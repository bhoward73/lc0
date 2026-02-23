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
//#define POSTERIOR_POLICY_ALGORITHM new_policy_common_ucb_Simple

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

//debug
int bound_test(float* x, int len_x, float bound) {
  int i;
  for(i=0;i<len_x;i++) {
    if(fabsf(x[i]) > bound) return 0;
  }
  return 1;
}

void assign_simulations(int* chunksizes, int budget, float* pi, int n) {
  assert(n > 0);
  if (budget <= 0) {
    memset(chunksizes, 0, n * sizeof(int));
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
    memset(chunksizes,0,n*sizeof(int));
    while((count < budget) && (x < budget) && (i<n)) {
      //printf("x = %.2f, s = %.2f, i = %d\n",x,s,i);
      if(x < s) {
        chunksizes[i]++;
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

// simpler version which is simpler than Newton; based on asymptotic approximation
// but is not equivalent (it essentially makes the c constant adjustable)
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

void legalize_policy(float* pi_legal, float* pi, float* g) {
  int n = numActions();
  int num_valid_actions;
  std::vector<int> actions(n);
  int i;
  num_valid_actions = getValidActions(actions.data(), g);
  assert(num_valid_actions > 0);
  memset(pi_legal, 0, n*sizeof(float));
  for(i=0;i<num_valid_actions;i++) {
    pi_legal[actions[i]] = pi[actions[i]];
  }
  float sum_pi = sum_of_float_array(pi_legal, n);
  if (!std::isfinite(sum_pi) || sum_pi <= 0.0f) {
    const float uniform = 1.0f / static_cast<float>(num_valid_actions);
    for (i = 0; i < num_valid_actions; i++) {
      pi_legal[actions[i]] = uniform;
    }
  } else {
    for(i=0;i<n;i++) pi_legal[i] /= sum_pi;
  }
}



// all the array data is allocated in python
// partly because we want to see this data in python
// we assume that the policy and value 
// for the root states are already computed
// and sitting in the policy and value arrays
// we also assume that the root states are sitting in the G array
// hence the state should be that the inference_stack is empty
// but the new_stack is full of root states (only)
void* MCTS_init(int const num_lanes,  
				int const numSims,
        float const c_puct,
        float* new_policy,
				float* new_value, 
				float* G,
				float* policy,
				float* value,
				float* Q,
				float* N,
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
  int gamesize = gameLength();
  float* Q;
  float* N;

  assert(t->sims_remaining[parent] >= 1 + sims_child);
  player_id_parent = playerId(t->G + parent*gamesize);
  v = v_child * player_id_child * player_id_parent;
  Q = t->Q + parent*n;
  N = t->N + parent*n;
  Q[a0] = (Q[a0]*N[a0] + v*sims_child)/(N[a0] + sims_child);
  N[a0] += sims_child;
  t->sims_remaining[parent] -= sims_child;
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
  int T = t->sims[idx] - 1;

  v0 = t->value[idx]; // network value for this state
  pi0 = t->policy + idx*n; // should already be legalized
  Q = t->Q + idx*n;
  N = t->N + idx*n;

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

  int T = t->sims[idx] - 1;

  v0 = t->value[idx]; // network value for this state
  pi0 = t->policy + idx*n; // should already be legalized
  Q = t->Q + idx*n;
  N = t->N + idx*n;

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
  v += (v0-v)/(T+1);

  // finally compute the new policy pi1
  for(i=0;i<len_mask;i++) {
    pi1[mask[i]] = pi1_mask[i];
  }

  // copy the new policy and new value in the data array
  memcpy(t->new_policy + idx*n, pi1.data(), n*sizeof(float));
  t->new_value[idx] = v;

  return v;
}


/* a child state sends its own value v_child and 
number of simulations sims_child
to the parent.
parent updates Q[a0], N[a0], and sims (remaining).
if sims remaining of parent becomes 1,
then the parent moves up to grandparent, etc, 
until the sims > 1, or until reaching 
one of the original root states. */

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
    player_id_child = playerId(t->G + child*gameLength());
    a0 = t->a0[child];
    parent = t->parent[child];
    t->sims_remaining[child] = 0;
    sims_remaining = update_parent(t, parent, a0, v_child, sims_child, player_id_child);
  }
}


// completely flushes the new_stack
// and populates the inference_stack with whatever inferences are required
void MCTS_flush_new_stack(void* const mcts)
{
  MCTS_new_t* t = (MCTS_new_t*) mcts;
  // first I will do this without dictionaries
  // which is really only correct for nonrandom games
  // each action is taken a certain number of times
  // but I will assume that whichever state appears 
  // the first time will be repeated the same number of times
  // as the action multiplicity
  int i0;
  // int i;
  int numSims;
  int p;
  int a0,a;
  float v0, v_child;
  float player_g, player_h;
  float* g;
  float* pi0;
  int gamesize = gameLength();
  int n = numActions();
  const int capacity = t->num_lanes * t->numSims;
  std::vector<float> h(gamesize);
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
    g = t->G + i0*gamesize;
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
    for(a=0;a<n;a++) {
      if(action_counts[a] == 0) continue;
      nextState(h.data(), g, a);
      player_h = playerId(h.data());
      ended = gameEnded(&score, h.data());
      if(ended) {
        v_child = score * player_h;
        pending.push_back({i0, a, v_child, action_counts[a], player_h, -1});
        continue;
      }
      m = t->row_count[0];
      assert(m >= 0 && m < capacity);
      assert(*(t->inference_stack_size) >= 0 && *(t->inference_stack_size) < capacity);
      memcpy(t->G + m*gamesize, h.data(), gamesize*sizeof(float));
      t->parent[m] = i0;
      t->a0[m] = a;
      t->sims[m] = action_counts[a];
      t->sims_remaining[m] = action_counts[a];
      t->inference_stack[*(t->inference_stack_size)] = m;
      t->inference_stack_size[0]++;
      t->row_count[0]++;
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


