# HiveMind (Swarm/Fusion) Implementation

## Phase 1: Core Infrastructure
- [/] Design and Plan
    - [x] Explore codebase
    - [x] Create comprehensive implementation plan
- [ ] Create `src/rotator_library/ensemble_manager.py`
    - [ ] Define `EnsembleManager` class skeleton
    - [ ] Implement config loading and validation
    - [ ] Implement `is_ensemble()` detection
    - [ ] Implement conflict resolution for naming
- [ ] Modify `src/rotator_library/client.py`
    - [ ] Initialize `EnsembleManager` in `__init__`
    - [ ] Integrate into `acompletion()` dispatcher
    - [ ] Add logging for HiveMind operations
- [ ] Create `ensemble_config.json`
    - [ ] Define schema for Fusions
    - [ ] Define schema for Swarm defaults
    - [ ] Define arbitration strategies

## Phase 2: Basic Swarm Mode
- [ ] Implement Swarm Features
    - [ ] `_prepare_drones()` - basic cloning
    - [ ] `_execute_parallel()` - asyncio.gather
    - [ ] `_format_for_arbiter()` - response aggregation
    - [ ] `_build_arbiter_prompt()` - synthesis strategy
    - [ ] `_call_arbiter()` - judge execution
- [ ] Testing
    - [ ] Test basic 3-drone swarm
    - [ ] Test arbiter synthesis
    - [ ] Test partial failures

## Phase 3: Advanced Swarm Features
- [ ] Temperature Jitter
    - [ ] Implement jitter logic
    - [ ] Test randomness and clamping
- [ ] Adversarial Mode
    - [ ] Implement adversarial prompt injection
    - [ ] Test with configurable count
- [ ] Blind Switch
    - [ ] Implement response anonymization
    - [ ] Test with blind=true/false
- [ ] Confidence Scoring
    - [ ] Implement score extraction
    - [ ] Add logging for scores

## Phase 4: Fusion Mode
- [ ] Implement Fusion Features
    - [ ] `_prepare_models()` - multi-model setup
    - [ ] Role assignment and prompts
    - [ ] Role context for Arbiter
    - [ ] Weight system (future)
- [ ] Testing
    - [ ] Test 2-model fusion
    - [ ] Test role context injection
    - [ ] Test specialist descriptions

## Phase 5: Recursive/Reflective Mode
- [ ] Implement Recursion
    - [ ] Consensus check logic
    - [ ] Conflict extraction
    - [ ] `_trigger_round_2()` implementation
    - [ ] Max rounds enforcement
- [ ] Testing
    - [ ] Test low-confidence trigger
    - [ ] Test Round 2 critique
    - [ ] Test final re-synthesis

## Phase 6: Polish & Edge Cases
- [ ] Error Handling
    - [ ] Partial failure handling
    - [ ] Arbiter failure fallback
    - [ ] Infinite recursion prevention
- [ ] Performance
    - [ ] Latency logging
    - [ ] Token usage tracking
    - [ ] Rate limit mitigation
- [ ] Documentation
    - [ ] User guide
    - [ ] Example configs
    - [ ] API reference

## Verification
- [ ] Automated Tests
    - [ ] test_ensemble_manager.py (all 8 test cases)
    - [ ] test_swarm_logic.py
    - [ ] test_fusion_logic.py
    - [ ] test_recursion.py
- [ ] Manual Tests
    - [ ] Scenario 1: Simple Swarm
    - [ ] Scenario 2: Adversarial Swarm
    - [ ] Scenario 3: Fusion with Roles
    - [ ] Scenario 4: Recursive Refinement
