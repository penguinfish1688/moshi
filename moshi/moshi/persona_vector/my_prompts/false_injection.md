"""
## Robustness to false injection
  - Used to explain that random injection of vectors speaking does not affect the response quality
### CLI option --generate-input-wav
  - 
### CLI option --generate-random-vector
  - Generate steering vector just like in user_interrupt.py, extract from the SVM and times alpha
  - take --layer to determine which layer to inject into
  - take --prob to determine the probability of injection of each token
  - take --root-dir to generate the steering_vector.json under root_dir/*/steering_vector.json
  - So to decide what to write to steering_vector.json, for each tokens, with probability prob, write the vector to the json.

### (You don't have to bother this) Then we will run the source intr_inference_steer.py to generate output.wav that is steered

### After inference, the user will run this program again with --evaluate-results
  - Read the output.wav under root_dir/*/output.wav
  - run with prompt that evaluate whether how good the response is on 0-5 scale
    - 0 is totaly unrelated
    - 1 is not related
    - 2 is slightly related
    - 3 is related
    - 4 is highly related
    - 5 is perfectly related
  - Take --layer and --prob and --root-dir to know which results to evaluate
  - Save the results to root_dir/false_injection_{layer}_{prob}.json



"""