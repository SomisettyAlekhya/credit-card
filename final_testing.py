import pickle
import numpy as np
import sys
from log_file import phase_1

logger = phase_1("Final_testing")


def testing_():
    try:
        # Load the model
        with open("credit_card.pkl", "rb") as f:
            model_ = pickle.load(f)

        logger.info(f"Loaded model type: {type(model_)}")

        # Generate a random test sample (matching the training feature count)
        temp = np.random.random((1, 10))  # 1 sample with 10 features

        # Make a prediction
        prediction = model_.predict(temp)[0]

        if prediction == 0:
            logger.info("Bad Customer")
        else:
            logger.info("Good Customer")

        prediction = model_.predict(temp)[0]

        if prediction == 0:
            logger.info("Bad Customer")
        else:
            logger.info("Good Customer")

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")


# Run the test function
if __name__ == "__main__":
    testing_()
