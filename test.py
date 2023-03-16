from notebooks.setup import test_generator, model, latest_checkpoint
model.load_weights(latest_checkpoint)

import time
start = time.perf_counter()
model.evaluate(test_generator, return_dict=True)
total = time.perf_counter() - start
print(f"Took {total:.2f} seconds in total, {total / len(test_generator):.2f} seconds for each sample")

test_input, test_output, mr_targets, us_targets, _ = test_generator(10)
test_pred = model.predict(test_input)
print(test_generator.patients_cases[10])
# model.evaluate(test_input, test_output, return_dict=True)
