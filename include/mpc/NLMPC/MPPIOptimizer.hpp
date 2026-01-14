/*
 * Copyright (c) 2023-2026 Nicola Piccinelli
 * All rights reserved.
 */
#pragma once

#include <mpc/NLMPC/NLOptimizerBase.hpp>
#include <mpc/NLMPC/Objective.hpp>
#include <mpc/NLMPC/Constraints.hpp>
#include <mpc/NLMPC/Mapping.hpp>
#include <mpc/Logger.hpp>
#include <mpc/Types.hpp>

#include <random>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <thread>
#include <atomic>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <algorithm>

namespace mpc
{
    /**
     * MPPI optimizer - multi-threaded using a fixed thread-pool
     * - epsilon-based update with Boltzmann weights
     * - optional IT augmentation term
     * - AR(1) temporal correlation of noise
     * - clamping of inputs to specified bounds
     * - momentum (beta) on the update
     * - receding-horizon shift of the nominal
     * - optional sub-stepped Euler for continuous-time models
     */
    template <MPCSize sizer>
    class MPPIOptimizer : public NLOptimizerBase<sizer>
    {
    private:
        using IComponent<sizer>::checkOrQuit;

        using IDimensionable<sizer>::nx;
        using IDimensionable<sizer>::nu;
        using IDimensionable<sizer>::ny;
        using IDimensionable<sizer>::ph;
        using IDimensionable<sizer>::ch;
        using IDimensionable<sizer>::ineq;
        using IDimensionable<sizer>::eq;

        using NLOptimizerBase<sizer>::mapping;
        using NLOptimizerBase<sizer>::model;
        using NLOptimizerBase<sizer>::objFunc;
        using NLOptimizerBase<sizer>::conFunc;

        using NLOptimizerBase<sizer>::result;
        using NLOptimizerBase<sizer>::sequence;
        using NLOptimizerBase<sizer>::currentSlack;

        // ---------------------------------------------------------
        // Holds all buffers required for a single thread to avoid
        // dynamic allocation during the control loop.
        // ---------------------------------------------------------
        struct WorkerContext
        {
            size_t local_task_id = 0;
            std::mt19937 rng;
            std::normal_distribution<double> standard_normal_dist{0.0, 1.0};

            // Simulation buffers
            mat<(sizer.ph + 1), sizer.nx> Xbuf;
            mat<(sizer.ph + 1), sizer.nu> Ubuf;
            mat<(sizer.ph + 1), sizer.ny> Ybuf;

            // Noise and Control buffers
            mat<sizer.ch, sizer.nu> Eps;
            mat<sizer.ch, sizer.nu> Uroll;
        };

    public:
        MPPIOptimizer() = default;

        // Ensure threads are stopped cleanly on destruction
        ~MPPIOptimizer() override
        {
            shutdownThreadPool();
        }

        void onInit() override
        {
            checkOrQuit();

            // Allocate Main Optimizer Memory
            COND_RESIZE_MAT(sizer, U_mean, ch(), nu());
            U_mean.setZero();

            COND_RESIZE_MAT(sizer, x_lower, ph() + 1, nx());
            COND_RESIZE_MAT(sizer, x_upper, ph() + 1, nx());
            x_lower.setConstant(-std::numeric_limits<double>::infinity());
            x_upper.setConstant(+std::numeric_limits<double>::infinity());

            COND_RESIZE_MAT(sizer, u_lower, ph() + 1, nu());
            COND_RESIZE_MAT(sizer, u_upper, ph() + 1, nu());
            u_lower.setConstant(-std::numeric_limits<double>::infinity());
            u_upper.setConstant(+std::numeric_limits<double>::infinity());

            COND_RESIZE_MAT(sizer, WeightedEps, ch(), nu());

            // Buffers for the final "optimal" rollout
            COND_RESIZE_MAT(sizer, Xbuf_final, ph() + 1, nx());
            COND_RESIZE_MAT(sizer, Ubuf_final, ph() + 1, nu());
            COND_RESIZE_MAT(sizer, Ybuf_final, ph() + 1, ny());

            currentSlack = 0;
            cycle_idx = 0;
            is_first_iteration = true;

            // Initialize Persistent Thread Pool & Workspaces
            initializeThreadPool();

            Logger::instance().log(Logger::LogType::DETAIL)
                << "MPPI initialized (nx=" << nx()
                << ", nu=" << nu()
                << ", ph=" << ph()
                << ", ch=" << ch()
                << ", Threads=" << workers.size() << ")" << std::endl;
        }

        void setParameters(const Parameters &param) override
        {
            checkOrQuit();
            const auto mppi_param = dynamic_cast<const MPPIParameters *>(&param);
            if (mppi_param)
            {
                params = *mppi_param;
            }

            // Resize storage for costs and noise history based on num_rollouts
            if (S_list.size() != params.num_rollouts)
            {
                S_list.resize(params.num_rollouts);
                eps_storage.resize(params.num_rollouts);
                for (auto &e : eps_storage)
                {
                    COND_RESIZE_MAT(sizer, e, ch(), nu());
                }
            }

            Logger::instance().log(Logger::LogType::INFO)
                << "MPPI Parameters:" << std::endl
                << "  max iterations:       " << params.maximum_iteration << std::endl
                << "  num rollouts:         " << params.num_rollouts << std::endl
                << "  sigma:                " << params.sigma << std::endl
                << "  lambda:               " << params.lambda << std::endl
                << "  state bound penalty:  " << params.state_bound_penalty << std::endl
                << "  ineq penalty:         " << params.ineq_penalty << std::endl
                << "  barrier steepness:    " << params.barrier_steepness << std::endl
                << "  eq penalty:           " << params.eq_penalty << std::endl
                << "  beta (momentum):      " << params.beta << std::endl
                << "  rho (AR1 corr.):      " << params.rho << std::endl
                << "  integration substeps: " << params.integration_substeps << std::endl;
        }

        bool bindObjective() override
        {
            checkOrQuit();
            return true;
        }

        bool bindEq(ConstraintsType, const cvec<(sizer.ph * sizer.nx)>) override
        {
            checkOrQuit();
            return true;
        }

        bool bindUserIneq(ConstraintsType, const cvec<sizer.ineq>) override
        {
            checkOrQuit();
            return true;
        }
        
        bool bindUserEq(ConstraintsType, const cvec<sizer.eq>) override
        {
            checkOrQuit();
            return true;
        }

        bool setStateBounds(const cvec<sizer.nx> &lower_bounds,
                            const cvec<sizer.nx> &upper_bounds,
                            const HorizonSlice &slice) override
        {
            checkOrQuit();

            size_t start = (slice.start == -1) ? 0 : static_cast<size_t>(slice.start);
            size_t end_t = (slice.end == -1) ? (ph() + 1) : static_cast<size_t>(slice.end);
            end_t = std::min(end_t, static_cast<size_t>(ph() + 1));

            for (size_t t = start; t < end_t; ++t)
            {
                x_lower.row(t) = lower_bounds.transpose();
                x_upper.row(t) = upper_bounds.transpose();
            }

            return true;
        }

        bool setInputBounds(const cvec<sizer.nu> &lower_bounds,
                            const cvec<sizer.nu> &upper_bounds,
                            const HorizonSlice &slice) override
        {
            checkOrQuit();

            size_t start = (slice.start == -1) ? 0 : static_cast<size_t>(slice.start);
            size_t end_t = (slice.end == -1) ? (ph() + 1) : static_cast<size_t>(slice.end);
            end_t = std::min(end_t, static_cast<size_t>(ph() + 1));

            for (size_t t = start; t < end_t; ++t)
            {
                u_lower.row(t) = lower_bounds.transpose();
                u_upper.row(t) = upper_bounds.transpose();
            }

            return true;
        }

        void run(const cvec<sizer.nx> &x0, const cvec<sizer.nu> &u0) override
        {
            using namespace std::chrono;
            auto t0 = high_resolution_clock::now();

            checkOrQuit();

            if (is_first_iteration || !params.enable_warm_start || !result.is_feasible)
            {
                // we initialize the mean from u0 only if:
                // - it's the first iteration
                // - warm start is disabled
                // - last solution was not feasible
                initializeMeanFrom(u0);
            }

            Result<sizer.nu> r;
            const size_t N = params.num_rollouts;

            // Pre-calculate AR(1) constants
            const double rho = params.rho;
            const double noise_coef = std::sqrt(std::max(1.0 - rho * rho, 0.0));

            for (size_t it = 0; it < params.maximum_iteration; ++it)
            {
                WeightedEps.setZero();
                double w_sum = 0.0;

                // -------------------------------------------------------------
                // PARALLEL EXECUTION
                // -------------------------------------------------------------
                // Reset atomic counter for this iteration
                next_rollout_idx.store(0, std::memory_order_relaxed);
                active_workers_count.store(workers.size(), std::memory_order_relaxed);

                // Wake up workers
                {
                    std::lock_guard<std::mutex> lock(pool_mutex);
                    // Update shared state for workers
                    current_x0 = x0;
                    current_noise_coef = noise_coef;
                    current_iter = it;

                    pool_task_id++;
                }
                pool_cv.notify_all();

                // Wait for all workers to finish their batch
                {
                    std::unique_lock<std::mutex> lock(pool_mutex);
                    pool_done_cv.wait(lock, [this]() { return active_workers_count.load(std::memory_order_relaxed) == 0; });
                }

                // -------------------------------------------------------------
                // REDUCTION & UPDATE (Single Threaded)
                // -------------------------------------------------------------
                double S_best = *std::min_element(S_list.begin(), S_list.end());

                // Numerically stable softmax
                double log_Z = 0.0;
                for (size_t k = 0; k < N; ++k)
                {
                    log_Z += std::exp((S_best - S_list[k]) / params.lambda);
                }

                const double inv_log_Z = (log_Z > 1e-9) ? (1.0 / log_Z) : 0.0;

                for (size_t k = 0; k < N; ++k)
                {
                    const double w = std::exp((S_best - S_list[k]) / params.lambda) * inv_log_Z;
                    WeightedEps += w * eps_storage[k];
                }

                // Momentum update
                U_mean += params.beta * WeightedEps;
            }

            // -------------------------------------------------------------
            // FINAL TRAJECTORY & FEASIBILITY CHECK
            // -------------------------------------------------------------

            // Clamp final mean to be safe
            mat<sizer.ch, sizer.nu> U_mean_clamped = U_mean;
            for (size_t t = 0; t < ch(); ++t)
            {
                for (size_t j = 0; j < nu(); ++j)
                {
                    U_mean_clamped(t, j) = std::clamp(U_mean(t, j), u_lower(t, j), u_upper(t, j));
                }
            }

            r.cmd = U_mean_clamped.row(0);

            // Simulate one last time to populate the result struct
            simulateClosedLoop(x0, U_mean_clamped, Xbuf_final, Ubuf_final);
            Ybuf_final = model->getOutput(Xbuf_final, Ubuf_final);
            r.cost = evaluateRolloutOnce(x0, U_mean_clamped, Xbuf_final, Ubuf_final, Ybuf_final);

            // Check feasibility
            const auto z = mapping->wrapVector(Xbuf_final, Ubuf_final, currentSlack);
            r.is_feasible = conFunc ? conFunc->isFeasible(z) : true;
            bool bounds_satisfied = checkBounds(Xbuf_final, Ubuf_final, r);

            r.is_feasible = r.is_feasible && bounds_satisfied;
            // MPPI is a sampling method, it always returns a result
            r.status = ResultStatus::SUCCESS;

            sequence.state = Xbuf_final;
            sequence.input = Ubuf_final;
            sequence.output = Ybuf_final;
            result = r;

            // Receding horizon shift for next cycle
            shiftNominal();
            is_first_iteration = false;
            cycle_idx++;

            auto dt_us = duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
            Logger::instance().log(Logger::LogType::DETAIL)
                << "MPPI done: cost=" << r.cost << ", time(us)=" << dt_us << std::endl;
        }

    private:
        // ---------------------------------------------------------
        // THREAD POOL IMPLEMENTATION
        // ---------------------------------------------------------
        void initializeThreadPool()
        {
            // Clean up if re-initializing
            shutdownThreadPool();

            size_t num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
            workers.reserve(num_threads);
            worker_contexts.resize(num_threads);

            pool_running = true;
            pool_task_id = 0;

            for (size_t i = 0; i < num_threads; ++i)
            {
                // Pre-allocate thread-local buffers
                COND_RESIZE_MAT(sizer, worker_contexts[i].Xbuf, ph() + 1, nx());
                COND_RESIZE_MAT(sizer, worker_contexts[i].Ubuf, ph() + 1, nu());
                COND_RESIZE_MAT(sizer, worker_contexts[i].Ybuf, ph() + 1, ny());
                COND_RESIZE_MAT(sizer, worker_contexts[i].Eps, ch(), nu());
                COND_RESIZE_MAT(sizer, worker_contexts[i].Uroll, ch(), nu());

                worker_contexts[i].local_task_id = 0;

                // Spawn thread
                workers.emplace_back(&MPPIOptimizer::workerLoop, this, i);
            }
        }

        void shutdownThreadPool()
        {
            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                pool_running = false;
                pool_task_id++;
            }

            pool_cv.notify_all();

            for (auto &t : workers)
            {
                if (t.joinable())
                {
                    t.join();
                }
            }
            workers.clear();
            worker_contexts.clear();
        }

        void workerLoop(size_t thread_id)
        {
            auto &ctx = worker_contexts[thread_id];

            while (true)
            {
                // Wait for signal
                {
                    std::unique_lock<std::mutex> lock(pool_mutex);
                    pool_cv.wait(lock, [this, &ctx] { return (pool_task_id > ctx.local_task_id) || !pool_running; });

                    if (!pool_running)
                    {
                        return;
                    }

                    ctx.local_task_id = pool_task_id;
                }

                // Process rollouts
                size_t k;
                while ((k = next_rollout_idx.fetch_add(1, std::memory_order_relaxed)) < params.num_rollouts)
                {
                    // Seed based on K (rollout index)
                    std::seed_seq seq{
                        params.user_seed.value_or(123456789u),
                        static_cast<uint32_t>(cycle_idx.load(std::memory_order_relaxed)),
                        static_cast<uint32_t>(current_iter),
                        static_cast<uint32_t>(k)};

                    ctx.rng.seed(seq);
                    ctx.standard_normal_dist.reset();

                    // 1. Generate Noise
                    ctx.Eps.setZero();

                    for (size_t j = 0; j < nu(); ++j)
                    {
                        double eps_prev = 0.0;
                        const double sigma_j = (j < params.sigma.size()) ? params.sigma(j) : 1.0;

                        for (size_t t = 0; t < ch(); ++t)
                        {
                            const double eta = ctx.standard_normal_dist(ctx.rng) * sigma_j;
                            const double eps = params.rho * eps_prev + current_noise_coef * eta;
                            ctx.Eps(t, j) = eps;
                            eps_prev = eps;
                        }
                    }

                    // 2. Compute Clamped Control
                    for (size_t t = 0; t < ch(); ++t)
                    {
                        for (size_t j = 0; j < nu(); ++j)
                        {
                            double v = U_mean(t, j) + ctx.Eps(t, j);
                            v = std::clamp(v, u_lower(t, j), u_upper(t, j));
                            ctx.Uroll(t, j) = v;
                        }
                    }

                    // 3. Evaluate Cost (Thread-Safe)
                    // We write directly to global vectors at index k. Since k is unique, no locking needed.
                    S_list[k] = evaluateRolloutOnce(current_x0, ctx.Uroll, ctx.Xbuf, ctx.Ubuf, ctx.Ybuf);
                    eps_storage[k] = ctx.Eps; // Store noise for weight calculation
                }

                // Decrement active counter. If last worker, notify main thread.
                if (active_workers_count.fetch_sub(1, std::memory_order_relaxed) == 1)
                {
                    pool_done_cv.notify_one();
                }
            }
        }

        // ---------------------------------------------------------
        // HELPERS
        // ---------------------------------------------------------
        bool checkBounds(const mat<(sizer.ph + 1), sizer.nx> &X,
                         const mat<(sizer.ph + 1), sizer.nu> &U,
                         Result<sizer.nu> &r)
        {
            const double tolerance = 1e-4;
            int state_vio = -1, input_vio = -1;
            bool ok = true;

            for (size_t t = 0; t <= ph(); ++t)
            {
                // Check State Bounds
                for (size_t i = 0; i < nx(); ++i)
                {
                    if (X(t, i) < x_lower(t, i) - tolerance || X(t, i) > x_upper(t, i) + tolerance)
                    {
                        ok = false;
                        state_vio = static_cast<int>(i);
                        break;
                    }
                }

                // Check Input Bounds
                if (t < ch())
                {
                    for (size_t j = 0; j < nu(); ++j)
                    {
                        if (U(t, j) < u_lower(t, j) - tolerance || U(t, j) > u_upper(t, j) + tolerance)
                        {
                            ok = false;
                            input_vio = static_cast<int>(j);
                            break;
                        }
                    }
                }
                if (!ok)
                    break;
            }

            if (!ok)
            {
                r.solver_status_msg = "Bounds violation: " +
                                      (state_vio >= 0 ? "State " + std::to_string(state_vio) : "Input " + std::to_string(input_vio));
            }
            else
            {
                r.solver_status_msg = "Success";
            }
            return ok;
        }

        void initializeMeanFrom(const cvec<sizer.nu> &u0)
        {
            for (size_t i = 0; i < ch(); ++i)
            {
                for (size_t j = 0; j < nu(); ++j)
                {
                    U_mean(i, j) = std::clamp((double)u0(j), u_lower(i, j), u_upper(i, j));
                }
            }
        }

        void simulateClosedLoop(const cvec<sizer.nx> &x0,
                                const mat<sizer.ch, sizer.nu> &Uch,
                                mat<(sizer.ph + 1), sizer.nx> &X,
                                mat<(sizer.ph + 1), sizer.nu> &U)
        {
            X.row(0) = x0.transpose();
            for (size_t t = 0; t < ph(); ++t)
            {
                cvec<sizer.nu> u = (t < ch()) ? Uch.row(t).transpose() : Uch.row(ch() - 1).transpose();
                U.row(t) = u.transpose();

                cvec<sizer.nx> x_next;
                COND_RESIZE_CVEC(sizer, x_next, nx());

                if (model->isContinuousTime)
                {
                    const double dt = model->sampleTime / static_cast<double>(params.integration_substeps);
                    cvec<sizer.nx> x_curr = X.row(t).transpose();

                    // Simple sub-stepped Euler
                    for (int s = 0; s < params.integration_substeps; ++s)
                    {
                        cvec<sizer.nx> dx;
                        COND_RESIZE_CVEC(sizer, dx, nx());
                        model->vectorField(dx, x_curr, u, t);
                        x_curr += dt * dx;
                    }
                    x_next = x_curr;
                }
                else
                {
                    model->vectorField(x_next, X.row(t).transpose(), u, t);
                }
                X.row(t + 1) = x_next.transpose();
            }
            U.row(ph()) = U.row(ph() - 1); // Fill last U for completeness
        }

        double evaluateRolloutOnce(const cvec<sizer.nx> &x0,
                                   const mat<sizer.ch, sizer.nu> &Uch,
                                   mat<(sizer.ph + 1), sizer.nx> &Xbuf_arg,
                                   mat<(sizer.ph + 1), sizer.nu> &Ubuf_arg,
                                   mat<(sizer.ph + 1), sizer.ny> &Ybuf_arg)
        {
            simulateClosedLoop(x0, Uch, Xbuf_arg, Ubuf_arg);
            Ybuf_arg = model->getOutput(Xbuf_arg, Ubuf_arg);

            // Wrap vector for cost functions
            const auto z = mapping->wrapVector(Xbuf_arg, Ubuf_arg, currentSlack);
            double J = objFunc->evaluateCostThreadSafe(z);

            // Constraints handling
            if (conFunc)
            {
                if constexpr (sizer.ineq.value != 0)
                {
                    auto res = conFunc->evaluateIneqThreadSafe(z);
                    for (int i = 0; i < res.value.size(); ++i)
                    {
                        if (res.value(i) > 0)
                        {
                            J += params.ineq_penalty * std::exp(params.barrier_steepness * res.value(i));
                        }
                    }
                }
                if constexpr (sizer.eq.value != 0)
                {
                    auto res = conFunc->evaluateEqThreadSafe(z);
                    for (int i = 0; i < res.value.size(); ++i)
                    {
                        J += params.eq_penalty * res.value(i) * res.value(i);
                    }
                }
            }

            // State Bound Penalties
            for (size_t t = 0; t <= ph(); ++t)
            {
                for (size_t i = 0; i < nx(); ++i)
                {
                    double x = Xbuf_arg(t, i);
                    if (x < x_lower(t, i))
                    {
                        J += params.state_bound_penalty * std::exp(params.barrier_steepness * (x_lower(t, i) - x));
                    }
                    else if (x > x_upper(t, i))
                    {
                        J += params.state_bound_penalty * std::exp(params.barrier_steepness * (x - x_upper(t, i)));
                    }
                }
            }
            return J;
        }

        void shiftNominal()
        {
            if (ch() <= 1)
            {
                return;
            }

            for (size_t t = 0; t + 1 < ch(); ++t)
            {
                U_mean.row(t) = U_mean.row(t + 1);
            }

            U_mean.row(ch() - 1) = U_mean.row(ch() - 2);
        }

    private:
        MPPIParameters params{};
        mat<sizer.ch, sizer.nu> U_mean;

        // Bounds
        mat<(sizer.ph + 1), sizer.nx> x_lower, x_upper;
        mat<(sizer.ph + 1), sizer.nu> u_lower, u_upper;

        // Final Result Buffers (Main Thread Only)
        mat<(sizer.ph + 1), sizer.nx> Xbuf_final;
        mat<(sizer.ph + 1), sizer.nu> Ubuf_final;
        mat<(sizer.ph + 1), sizer.ny> Ybuf_final;

        // Shared Data (Workers Write, Main Reads)
        mat<sizer.ch, sizer.nu> WeightedEps;
        std::vector<double> S_list;
        std::vector<mat<sizer.ch, sizer.nu>> eps_storage;

        // Thread Pool Internals
        std::vector<std::thread> workers;
        std::vector<WorkerContext> worker_contexts;

        std::atomic<bool> pool_running{false};
        size_t pool_task_id{0};
        std::mutex pool_mutex;
        std::condition_variable pool_cv;
        std::condition_variable pool_done_cv;

        std::atomic<size_t> next_rollout_idx{0};
        std::atomic<size_t> active_workers_count{0};

        // Shared Context for Current Iteration
        cvec<sizer.nx> current_x0;
        double current_noise_coef{0.0};
        size_t current_iter{0};

        bool is_first_iteration = true;
        std::atomic<uint64_t> cycle_idx{0};
    };
} // namespace mpc