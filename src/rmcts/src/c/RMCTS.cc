/*
Copyright (c) 2025, Institute for Defense Analyses, 730 Glebe Rd, Alexandria, VA 22305-3086; 703-845-2500

This material may be reproduced by or for the U.S. Government pursuant to all applicable FAR and DFARS clauses.
*/

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "RMCTS.h"
#include "random.h"
#include "game.h"

#define SOFTPOWER 16.0
#define UCB_EPSILON 0.1

#define POSTERIOR_POLICY_ALGORITHM new_policy_common_ucb_Newton

bool rmcts_debug_root_enabled() {
  static int initialized = 0;
  static bool enabled = false;
  if (!initialized) {
    const char* env = std::getenv("RMCTS_DEBUG_ROOT");
    enabled = env && env[0] && env[0] != '0';
    initialized = 1;
  }
  return enabled;
}

int posterior_effective_T_from_N(const float* N, int n) {
  float sum_N = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum_N += N[i];
  }
  if (!std::isfinite(sum_N) || sum_N <= 0.0f) return 1;
  const int T = static_cast<int>(std::lround(sum_N));
  return (T > 0) ? T : 1;
}

float sum_of_float_array(float* a, int len_a) {
  int i;
  float s = 0.0;
  for(i=0;i<len_a;i++) s += a[i];
  return s;
}

int argmax_of_float_array(float* a, int len_a) {
  int i;
  float m;
  int i_max; 

  assert(len_a > 0); 
  m = a[0];
  i_max = 0;
  for(i=1;i<len_a;i++) {
    if(a[i]>m) {
      m = a[i];
      i_max = i;
    }
  }
  return i_max;
}

int softmax_of_float_array(float* a, int len_a) {
  int i, i_max;
  float m;
  if(len_a <= 0) return -1;
  std::vector<float> p(len_a);
  i_max = argmax_of_float_array(a, len_a);
  m = a[i_max];
  for(i=0;i<len_a;i++) {
    p[i] = exp(SOFTPOWER*(a[i]-m));
  }
  float sum_p = sum_of_float_array(p.data(), len_a);
  for(i=0;i<len_a;i++) {
    p[i] /= sum_p;
  }
  i_max = random_index(p.data(), len_a);
  return i_max;
}

int maximum_of_int_array(int* a, int len_a) {
  int i;
  int m;
  if(len_a==0) return 0;
  m = a[0];
  for(i=1;i<len_a;i++) if(a[i] > m) m = a[i];
  return m;
}

int bound_test(float* x, int len_x, float bound) {
  int i;
  for(i=0;i<len_x;i++) {
    if(fabsf(x[i]) > bound) return 0;
  }
  return 1;
}

void assign_simulations(int* sims_per_action, int budget, float* pi, int n) {
  assert(n > 0);
  if (budget <= 0) {
    memset(sims_per_action, 0, n * sizeof(int));
    return;
  }
  float x; // random number in [0,1)
  float s; // cumulative sum of pi*budget
  float sum_pi; // first ensure that pi is normalized
  int i,count;
  sum_pi = sum_of_float_array(pi, n);
  if (!std::isfinite(sum_pi) || sum_pi <= 0.0f) {
    const float uniform = 1.0f / static_cast<float>(n);
    for(i=0;i<n;i++) pi[i] = uniform;
  } else {
    for(i=0;i<n;i++) pi[i] /= sum_pi;
  }
  do {
    x = Knuth_drand();
    s = pi[0]*budget;
    i = 0;
    count = 0;
    memset(sims_per_action,0,n*sizeof(int));
    while((count < budget) && (x < budget) && (i<n)) {
      //printf("x = %.2f, s = %.2f, i = %d\n",x,s,i);
      if(x < s) {
        sims_per_action[i]++;
        x += 1.0;
        count++;
      } 
      else {
        i++;
        if (i < n) {
          s += pi[i]*budget;
        }
      }
    }
  } while(count < budget);
}


void new_policy_common_ucb_Newton(float* pi1, int n, float* Q, float c, float* pi0, int T) {
  double c0 = ((double) c) / sqrt((double) T);
  double delta, new_delta;
  double epsilon = 1.0e-12;
  double f, fprime, x;
  int i;
  float sum_pi;
  double Q_max;
  int a_max;

  float pi0_min = INFINITY;
  for(i=0;i<n;i++) {
    if(pi0[i] < pi0_min) pi0_min = pi0[i];
  }
  assert(pi0_min > 0.0);

  sum_pi = sum_of_float_array(pi0, n);
  for(i=0;i<n;i++) pi0[i] /= sum_pi;

  Q_max = -INFINITY;
  a_max = 0;
  for(i=0;i<n;i++) {
    if(((double) Q[i]) > Q_max) {
      Q_max = (double) Q[i];
      a_max = i;
    }
  }

  delta = c0 * ((double) pi0[a_max]);
  if(delta < epsilon) {
    delta = epsilon;
  }

  f = INFINITY;
  while(f > epsilon) {
    f = 0.0;
    fprime = 0.0;
    for(i=0;i<n;i++) {
      x = (c0 * ((double) pi0[i])) / ((Q_max - ((double) Q[i])) + delta);
      f += x;
      fprime -= x/((Q_max - ((double) Q[i])) + delta);
    }
    f -= 1.0;
    if(f <= 0.0) break;
    new_delta = delta - f/fprime;
    if(new_delta <= delta) break;
    delta = new_delta;
  }

  for(i=0;i<n;i++) {
    pi1[i] = (float) (c0 * ((double) pi0[i])) / ((Q_max - ((double) Q[i])) + delta);
  }
  sum_pi = sum_of_float_array(pi1, n);
  for(i=0;i<n;i++) pi1[i] /= sum_pi;
}

// Alternative posterior policy update (kept for experimentation).
void new_policy_common_ucb_Simple(float* pi1, int n, float* Q, float c, float* pi0, int T) {
  int i;
  float c0 = c / sqrt((float) T);
  float sum_pi1;
  float Qmax;
  float u;

  Qmax = -INFINITY;
  for(i=0;i<n;i++) {
    if(Q[i] > Qmax) Qmax = Q[i];
  }
  u = Qmax + c0;
  for(i=0;i<n;i++) {
    pi1[i] = c0*pi0[i] / (u - Q[i]);
  }
  sum_pi1 = sum_of_float_array(pi1, n);
  for(i=0;i<n;i++) {
    pi1[i] /= sum_pi1;
  }
} 

void legalize_policy(float* pi_legal, float* pi, GameStateHandle g) {
  int n = numActions();
  int num_valid_actions;
  std::vector<int> actions(n);
  int i;
  num_valid_actions = getValidActions(actions.data(), g);
  assert(num_valid_actions > 0);
  memset(pi_legal, 0, n*sizeof(float));

  // Build the legal-only prior and compute Z0 = sum_{a in L} pi0(a).
  // Then mix normalized legal prior with uniform on L, using Z0 as the
  // trust weight: pi0_legal = Z0 * pi0_L + (1-Z0) * U_L.
  float z0 = 0.0f;
  for (i = 0; i < num_valid_actions; i++) {
    const int a = actions[i];
    const float p = std::max(0.0f, pi[a]);
    pi_legal[a] = p;
    z0 += p;
  }

  if (!std::isfinite(z0) || z0 <= 0.0f) {
    const float uniform = 1.0f / static_cast<float>(num_valid_actions);
    for (i = 0; i < num_valid_actions; i++) {
      pi_legal[actions[i]] = uniform;
    }
    return;
  }

  const float z0_clamped = std::min(1.0f, std::max(0.0f, z0));
  const float uniform = 1.0f / static_cast<float>(num_valid_actions);
  for (i = 0; i < num_valid_actions; i++) {
    const int a = actions[i];
    const float pi0_l = pi_legal[a] / z0;
    pi_legal[a] = z0_clamped * pi0_l + (1.0f - z0_clamped) * uniform;
  }
}

// Initialization assumes root rows are already populated by the caller:
// - root states in G
// - root policy/value in policy/value
// - inference stack empty
// - new stack seeded with root rows
void* MCTS_init(int const num_lanes,  
				int const numSims,
        float const c_puct,
        float* new_policy,
				float* new_value, 
        GameStateHandle* G,
				float* policy,
				float* value,
				float* Q,
				float* N,
        int32_t* child,
				int32_t* parent,
				int32_t* a0,
				int32_t* sims,
        int32_t* sims_remaining,
				int32_t* inference_stack,
        int32_t* inference_stack_size,
				int32_t* new_stack,
        int32_t* new_stack_size,
        int32_t* num_completed,
        int32_t* row_count)
{
  MCTS_new_t* t;
  t = (MCTS_new_t*) calloc(1,sizeof(MCTS_new_t));
  t->num_lanes = num_lanes;
  t->numSims = numSims;
  t->c_puct = c_puct;

  t->new_policy = new_policy;
  t->new_value = new_value;    
  t->G = G;
  t->policy = policy;
  t->value = value;
  t->Q = Q;
  t->N = N;
  t->child = child;
  
  t->parent = parent;
  t->a0 = a0;
  t->sims = sims;
  t->sims_remaining = sims_remaining;
  t->inference_stack = inference_stack;
  t->inference_stack_size = inference_stack_size;
  t->new_stack = new_stack;
  t->new_stack_size = new_stack_size;
  t->num_completed = num_completed;
  t->row_count = row_count;

  return (void*) t;
}

void MCTS_free(void* const mcts) {
  MCTS_new_t* t = (MCTS_new_t*) mcts;
  if(!t) return;
  free(t);
}

int update_parent(MCTS_new_t* const t, int parent, int a0, float v_child, int sims_child, float player_id_child){
  // updates Q and N values for the parent
  // returns the number of simulations remaining for the parent
  float v, player_id_parent;
  int n = numActions();
  float* Q;
  float* N;

  assert(t->sims_remaining[parent] >= 1 + sims_child);
  player_id_parent = playerId(t->G[parent]);
  v = v_child * player_id_child * player_id_parent;
  Q = t->Q + parent*n;
  N = t->N + parent*n;
  const float q_old = Q[a0];
  const float n_old = N[a0];
  Q[a0] = (Q[a0]*N[a0] + v*sims_child)/(N[a0] + sims_child);
  N[a0] += sims_child;
  t->sims_remaining[parent] -= sims_child;

  if (rmcts_debug_root_enabled() && parent < t->num_lanes) {
    std::printf(
        "RMCTS_ROOT_UPDATE a=%d sims_child=%d v_child=%.6f pid_child=%.1f pid_parent=%.1f "
        "v_used=%.6f Q_old=%.6f N_old=%.0f Q_new=%.6f N_new=%.0f sims_rem=%d\n",
        a0, sims_child, v_child, player_id_child, player_id_parent, v, q_old,
        n_old, Q[a0], N[a0], t->sims_remaining[parent]);
    std::fflush(stdout);
  }
  return t->sims_remaining[parent];
}

float compute_new_value_nonroot(MCTS_new_t* t, int idx) {
  // here we restrict to where N > 0 in computing the posterior policy
  // we only need to compute the final value here.
  assert(idx >= t->num_lanes);
  assert(t->sims_remaining[idx] == 1);
  assert(t->sims[idx] > 1);

  float v, v0;
  float* pi0;
  float* Q;
  float* N;
  int len_mask = 0;
  float new_portion = 0.0; // sum of pi0 over actions where N > 0
  float reciprocal_new_portion;
  int i,a;
  int n = numActions();

  v0 = t->value[idx]; // network value for this state
  pi0 = t->policy + idx*n; // should already be legalized
  Q = t->Q + idx*n;
  N = t->N + idx*n;
  int T = posterior_effective_T_from_N(N, n);

  std::vector<int> mask(n);
  std::vector<float> pi0_mask(n, 0.0f);
  std::vector<float> Q_mask(n, 0.0f);
  std::vector<float> pi1_mask(n);

  for(a=0;a<n;a++) {
    if(N[a] > 0) {
      mask[len_mask] = a;
      len_mask++;
    } 
  }

  for(i=0;i<len_mask;i++) {
    a = mask[i];
    assert(pi0[a] > 0.0);
    pi0_mask[i] = pi0[a];
    Q_mask[i] = Q[a];
    new_portion += pi0[a];
  }

  reciprocal_new_portion = 1.0/new_portion;
  for(i=0;i<len_mask;i++) {
    pi0_mask[i] *= reciprocal_new_portion;
  }

  POSTERIOR_POLICY_ALGORITHM(pi1_mask.data(), len_mask, Q_mask.data(), t->c_puct, pi0_mask.data(), T);
  // new_policy_common_ucb_Newton(pi1_mask, len_mask, Q_mask, t->c_puct, pi0_mask, T);

  // Posterior value is computed on measured actions only, using the
  // undampened posterior distribution pi1_mask over that set.
  v = 0.0;
  for(i=0;i<len_mask;i++) {
    v += pi1_mask[i]*Q_mask[i];
  }
  v += (v0-v)/(T+1);

  return v;
}

float compute_new_value_and_policy_root(MCTS_new_t* t, int idx) {
  // needs to learn both new policy and new value
  assert(idx < t->num_lanes);
  assert(t->sims_remaining[idx] == 1);
  assert(t->sims[idx] > 1);

  float v, v0;
  float* pi0; // should already be legalized
  float* Q;
  float* N;
  int len_mask = 0;
  float new_portion = 0.0; // sum of pi0 over actions where N > 0
  float reciprocal_new_portion;
  int i,a;
  int n = numActions();

  v0 = t->value[idx]; // network value for this state
  pi0 = t->policy + idx*n; // should already be legalized
  Q = t->Q + idx*n;
  N = t->N + idx*n;
  int T = posterior_effective_T_from_N(N, n);

  std::vector<int> mask(n);
  std::vector<float> pi0_mask(n, 0.0f);
  std::vector<float> Q_mask(n, 0.0f);
  std::vector<float> pi1_mask(n);
  std::vector<float> pi1(n, 0.0f);

  for(a=0;a<n;a++) {
    if(N[a] > 0) {
      mask[len_mask] = a;
      len_mask++;
    } 
  }

  for(i=0;i<len_mask;i++) {
    a = mask[i];
    assert(pi0[a] > 0.0);
    pi0_mask[i] = pi0[a];
    Q_mask[i] = Q[a];
    new_portion += pi0[a];
  }

  reciprocal_new_portion = 1.0/new_portion;

  for(i=0;i<len_mask;i++) {
    pi0_mask[i] *= reciprocal_new_portion;
  }

  POSTERIOR_POLICY_ALGORITHM(pi1_mask.data(), len_mask, Q_mask.data(), t->c_puct, pi0_mask.data(), T);
  //new_policy_common_ucb_Newton(pi1_mask, len_mask, Q_mask, t->c_puct, pi0_mask, T);

  v = 0.0;
  for(i=0;i<len_mask;i++) {
    v += pi1_mask[i]*Q_mask[i];
  }
  const float v_measured = v;
  v += (v0-v)/(T+1);

  if (rmcts_debug_root_enabled()) {
    std::printf(
        "RMCTS_ROOT_POSTERIOR idx=%d ZA=%.6f v_measured=%.6f v_final=%.6f v0=%.6f T=%d\n",
        idx, new_portion, v_measured, v, v0, T);
    std::fflush(stdout);
  }

  // finally compute the new policy pi1
  // Keep prior mass on actions outside A (where N[a] == 0).
  // On A, assign pi1_A scaled by ZA, where ZA = sum_{a in A} pi0[a].
  memcpy(pi1.data(), pi0, n * sizeof(float));
  for (i = 0; i < len_mask; i++) {
    pi1[mask[i]] = pi1_mask[i] * new_portion;
  }

  // copy the new policy and new value in the data array
  memcpy(t->new_policy + idx*n, pi1.data(), n*sizeof(float));
  t->new_value[idx] = v;

  return v;
}

// Propagates one completed child contribution upward; continues while each
// ancestor has exactly one simulation remaining and can be finalized.
void propagate(MCTS_new_t* t, int parent, int a0, float v_child, int sims_child, float player_id_child) 
{
  int child;
  int sims_remaining;

  sims_remaining = update_parent(t, parent, a0, v_child, sims_child, player_id_child);
  while(sims_remaining == 1) {
    if(parent < t->num_lanes) {
      compute_new_value_and_policy_root(t, parent);
      t->sims_remaining[parent] = 0;
      t->num_completed[0]++;
      break;
    }
    child = parent;
    v_child = compute_new_value_nonroot(t, child);
    sims_child = t->sims[child];
    player_id_child = playerId(t->G[child]);
    a0 = t->a0[child];
    parent = t->parent[child];
    t->sims_remaining[child] = 0;
    sims_remaining = update_parent(t, parent, a0, v_child, sims_child, player_id_child);
  }
}

// Two-phase flush:
// 1) expand current new_stack and queue backup events,
// 2) execute queued backups/propagation.
void MCTS_flush_new_stack(void* const mcts)
{
  MCTS_new_t* t = (MCTS_new_t*) mcts;
  int i0;
  // int i;
  int numSims;
  int p;
  int a0,a;
  float v0, v_child;
  float player_g, player_h;
  GameStateHandle g;
  float* pi0;
  int n = numActions();
  const int capacity = t->num_lanes * t->numSims;
  GameStateHandle h = 0;
  std::vector<float> pi0_legal(n);
  std::vector<int> action_counts(n);
  int m;
  int ended;
  float score;

  struct PendingPropagation {
    int parent;
    int action;
    float value;
    int sims;
    float player_id;
    int clear_idx;
  };
  std::vector<PendingPropagation> pending;

  if (*(t->new_stack_size) == 0) return;

  pending.reserve(*(t->new_stack_size) * 2);

  // Phase 1: Expand the tree and queue all propagations.
  // No recursive backup is done in this pass.

  while(*(t->new_stack_size) > 0) {
    i0 = t->new_stack[*(t->new_stack_size)-1];
    g = t->G[i0];
    t->new_stack_size[0]--;
    numSims = t->sims[i0];
    assert(numSims >= 1);
    p = t->parent[i0];
    a0 = t->a0[i0];
    v0 = t->value[i0];
    player_g = playerId(g);

    // legalize the prior policy pi0 in the lookup table
    pi0 = t->policy + i0*n;       
    legalize_policy(pi0_legal.data(), pi0, g);
    memcpy(t->policy + i0*n, pi0_legal.data(), n*sizeof(float));

    // if this is a leaf node, then propagate the value
    if(numSims == 1) {
      t->sims_remaining[i0] = 0;
      if (i0 < t->num_lanes) {
        memcpy(t->new_policy + i0*n, pi0_legal.data(), n*sizeof(float));
        t->new_value[i0] = v0;
        t->num_completed[0]++;
      } else {
        pending.push_back({p, a0, v0, 1, player_g, i0});
      }
      continue;
    }

    // not a leaf, so assigning action counts
    // and pushing the children onto the stack
    assign_simulations(action_counts.data(), numSims-1, pi0_legal.data(), n);

    if (rmcts_debug_root_enabled() && i0 < t->num_lanes) {
      std::printf("RMCTS_ROOT_ALLOC root=%d sims=%d\n", i0, numSims - 1);
      for (a = 0; a < n; a++) {
        if (action_counts[a] > 0) {
          std::printf("  a=%d count=%d prior=%.6f\n", a, action_counts[a],
                      pi0_legal[a]);
        }
      }
      std::fflush(stdout);
    }

    for(a=0;a<n;a++) {
      if(action_counts[a] == 0) continue;
      nextState(&h, g, a);
      player_h = playerId(h);
      ended = gameEnded(&score, h);
      if(ended) {
        v_child = score * player_h;
        pending.push_back({i0, a, v_child, action_counts[a], player_h, -1});
        continue;
      }
      const int child_idx = i0 * n + a;
      m = t->child[child_idx];
      if (m < 0) {
        m = t->row_count[0];
        assert(m >= 0 && m < capacity);
        assert(*(t->inference_stack_size) >= 0 && *(t->inference_stack_size) < capacity);
        t->child[child_idx] = m;
        t->G[m] = h;
        t->parent[m] = i0;
        t->a0[m] = a;
        t->sims[m] = action_counts[a];
        t->sims_remaining[m] = action_counts[a];
        t->inference_stack[*(t->inference_stack_size)] = m;
        t->inference_stack_size[0]++;
        t->row_count[0]++;
      } else {
        // Existing subtree row: assign only new sims for this chunk.
        t->sims[m] = action_counts[a];
        t->sims_remaining[m] = action_counts[a];
        assert(*(t->new_stack_size) >= 0 && *(t->new_stack_size) < capacity);
        t->new_stack[*(t->new_stack_size)] = m;
        t->new_stack_size[0]++;
      }
    }
  }

  // Phase 2: Execute queued propagations/backups.
  for (const auto& event : pending) {
    propagate(t, event.parent, event.action, event.value, event.sims, event.player_id);
    if (event.clear_idx >= 0) {
      t->sims_remaining[event.clear_idx] = 0;
    }
  }

}


