# # # # app.py
# # # from flask import Flask, request, render_template, jsonify
# # # import pickle
# # # import numpy as np

# # # app = Flask(__name__)

# # # # app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')


# # # # Load your trained ML model
# # # with open('model.pkl', 'rb') as file:
# # #     model = pickle.load(file)

# # # # Home route
# # # @app.route('/')
# # # def home():
# # #     return render_template('faqs.html')

# # # # Prediction route
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Get input data from the form
# # #         tensile_strength = float(request.form['tensile_strength'])
# # #         proof_stress = float(request.form['proof_stress'])
# # #         elongation = float(request.form['elongation'])

# # #         # Prepare input for the model
# # #         input_data = np.array([[tensile_strength, proof_stress, elongation]])

# # #         # Make prediction (assuming model outputs 1 = Good, 0 = Not Suitable)
# # #         prediction = model.predict(input_data)[0]
# # #         confidence = model.predict_proba(input_data).max() * 100  # Confidence score

# # #         # Generate result and suggestion
# # #         if prediction == 1:
# # #             result = "Good for Industry Use"
# # #             suggestion = "No improvements needed."
# # #         else:
# # #             result = "Not Suitable for Industry Use"
# # #             suggestion = "Consider increasing tensile strength or adjusting heat treatment."

# # #         return jsonify({
# # #             'result': result,
# # #             'confidence': f"{confidence:.2f}%",
# # #             'suggestion': suggestion,
# # #             'tensile_strength': tensile_strength,
# # #             'proof_stress': proof_stress,
# # #             'elongation': elongation
# # #         })
# # #     except Exception as e:
# # #         return jsonify({'error': str(e)}), 400

# # # if __name__ == '__main__':
# # #     app.run(debug=True)




# # # # from flask import Flask, request, render_template, jsonify
# # # # import pickle
# # # # import numpy as np

# # # # app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# # # # # Load trained ML model
# # # # with open('model.pkl', 'rb') as file:
# # # #     model = pickle.load(file)

# # # # # Home route
# # # # @app.route('/')
# # # # def home():
# # # #     return render_template('index.html')

# # # # # About route
# # # # @app.route('/about')
# # # # def about():
# # # #     return render_template('about.html')

# # # # # Properties route
# # # # @app.route('/properties')
# # # # def properties():
# # # #     return render_template('properties.html')

# # # # # Types route
# # # # @app.route('/types')
# # # # def types():
# # # #     return render_template('types.html')

# # # # # Applications route
# # # # @app.route('/applications')
# # # # def applications():
# # # #     return render_template('applications.html')

# # # # # FAQs route
# # # # @app.route('/faqs')
# # # # def faqs():
# # # #     return render_template('faqs.html')

# # # # # Resources route
# # # # @app.route('/resources')
# # # # def resources():
# # # #     return render_template('resources.html')

# # # # # Blog route
# # # # @app.route('/blogs')
# # # # def blog():
# # # #     return render_template('blog.html')

# # # # # Contact route
# # # # @app.route('/contact')
# # # # def contact():
# # # #     return render_template('contact.html')

# # # # # Prediction route
# # # # @app.route('/predict', methods=['POST'])
# # # # def predict():
# # # #     try:
# # # #         # Get input data from the form
# # # #         tensile_strength = float(request.form['tensile_strength'])
# # # #         proof_stress = float(request.form['proof_stress'])
# # # #         elongation = float(request.form['elongation'])

# # # #         # Prepare input for the model
# # # #         input_data = np.array([[tensile_strength, proof_stress, elongation]])

# # # #         # Make prediction (assuming model outputs 1 = Good, 0 = Not Suitable)
# # # #         prediction = model.predict(input_data)[0]
# # # #         confidence = model.predict_proba(input_data).max() * 100  # Confidence score

# # # #         # Generate result and suggestion
# # # #         if prediction == 1:
# # # #             result = "Good for Industry Use"
# # # #             suggestion = "No improvements needed."
# # # #         else:
# # # #             result = "Not Suitable for Industry Use"
# # # #             suggestion = "Consider increasing tensile strength or adjusting heat treatment."

# # # #         return jsonify({
# # # #             'result': result,
# # # #             'confidence': f"{confidence:.2f}%",
# # # #             'suggestion': suggestion,
# # # #             'tensile_strength': tensile_strength,
# # # #             'proof_stress': proof_stress,
# # # #             'elongation': elongation
# # # #         })
# # # #     except Exception as e:
# # # #         return jsonify({'error': str(e)}), 400

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)


# # from flask import Flask, request, render_template, jsonify
# # import pickle
# # import numpy as np
# # import logging

# # app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)

# # # Load trained ML model
# # try:
# #     with open('model.pkl', 'rb') as file:
# #         model = pickle.load(file)
# #     logger.info("Model loaded successfully")
# # except Exception as e:
# #     logger.error(f"Failed to load model: {str(e)}")
# #     raise

# # # Home route
# # @app.route('/')
# # def home():
# #     return render_template('index.html')

# # # About route
# # @app.route('/about')
# # def about():
# #     return render_template('about.html')

# # # Properties route
# # @app.route('/properties')
# # def properties():
# #     return render_template('properties.html')

# # # Types route
# # @app.route('/types')
# # def types():
# #     return render_template('types.html')

# # # Applications route
# # @app.route('/applications')
# # def applications():
# #     return render_template('applications.html')

# # # FAQs route
# # @app.route('/faqs')
# # def faqs():
# #     return render_template('faqs.html')

# # # Resources route
# # @app.route('/resources')
# # def resources():
# #     return render_template('resources.html')

# # # Blog route
# # @app.route('/blog')
# # def blog():
# #     return render_template('blog.html')

# # # Contact route
# # @app.route('/contact')
# # def contact():
# #     return render_template('contact.html')

# # # Prediction route
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     logger.debug("Received predict request")
# #     try:
# #         # Get input data from the form
# #         tensile_strength = float(request.form['tensile_strength'])
# #         proof_stress = float(request.form['proof_stress'])
# #         elongation = float(request.form['elongation'])

# #         # Validate inputs
# #         if tensile_strength <= 0 or proof_stress <= 0 or elongation <= 0:
# #             logger.warning("Invalid input: Negative or zero values")
# #             return jsonify({'error': 'All inputs must be positive numbers'}), 400

# #         # Prepare input for the model
# #         input_data = np.array([[tensile_strength, proof_stress, elongation]])
# #         logger.debug(f"Input data: {input_data}")

# #         # Make prediction
# #         prediction = model.predict(input_data)[0]
# #         confidence = model.predict_proba(input_data).max() * 100
# #         logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")

# #         # Generate result and suggestion
# #         if prediction == 1:
# #             result = "Good for Industry Use"
# #             suggestion = "No improvements needed."
# #         else:
# #             result = "Not Suitable for Industry Use"
# #             suggestion = "Consider increasing tensile strength or adjusting heat treatment."

# #         return jsonify({
# #             'result': result,
# #             'confidence': f"{confidence:.2f}%",
# #             'suggestion': suggestion,
# #             'tensile_strength': tensile_strength,
# #             'proof_stress': proof_stress,
# #             'elongation': elongation
# #         })
# #     except Exception as e:
# #         logger.error(f"Prediction error: {str(e)}")
# #         return jsonify({'error': str(e)}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # # # app.py (unchanged from previous response)
# # # from flask import Flask, request, render_template, jsonify
# # # import pickle
# # # import numpy as np
# # # import logging

# # # app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# # # logging.basicConfig(level=logging.DEBUG)
# # # logger = logging.getLogger(__name__)

# # # try:
# # #     with open('model.pkl', 'rb') as file:
# # #         model = pickle.load(file)
# # #     logger.info("Model loaded successfully")
# # # except Exception as e:
# # #     logger.error(f"Failed to load model: {str(e)}")
# # #     raise

# # # @app.route('/')
# # # def home():
# # #     return render_template('index.html')

# # # @app.route('/about')
# # # def about():
# # #     return render_template('about.html')

# # # @app.route('/properties')
# # # def properties():
# # #     return render_template('properties.html')

# # # @app.route('/types')
# # # def types():
# # #     return render_template('types.html')

# # # @app.route('/applications')
# # # def applications():
# # #     return render_template('applications.html')

# # # @app.route('/faqs')
# # # def faqs():
# # #     return render_template('faqs.html')

# # # @app.route('/resources')
# # # def resources():
# # #     return render_template('resources.html')

# # # @app.route('/blog')
# # # def blog():
# # #     return render_template('blog.html')

# # # @app.route('/contact')
# # # def contact():
# # #     return render_template('contact.html')

# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     logger.debug("Received predict request")
# # #     try:
# # #         tensile_strength = float(request.form['tensile_strength'])
# # #         proof_stress = float(request.form['proof_stress'])
# # #         elongation = float(request.form['elongation'])

# # #         if tensile_strength <= 0 or proof_stress <= 0 or elongation <= 0:
# # #             logger.warning("Invalid input: Negative or zero values")
# # #             return jsonify({'error': 'All inputs must be positive numbers'}), 400

# # #         input_data = np.array([[tensile_strength, proof_stress, elongation]])
# # #         logger.debug(f"Input data: {input_data}")

# # #         prediction = model.predict(input_data)[0]
# # #         confidence = model.predict_proba(input_data).max() * 100
# # #         logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")

# # #         if prediction == 1:
# # #             result = "Good for Industry Use"
# # #             suggestion = "No improvements needed."        
# # #         else:
# # #             result = "Not Suitable for Industry Use"
# # #             suggestion = "Increase annealing time and temperature to improve both tensile strength and ductility."

# # #         return jsonify({
# # #             'result': result,
# # #             'confidence': f"{confidence:.2f}%",
# # #             'suggestion': suggestion,
# # #             'tensile_strength': tensile_strength,
# # #             'proof_stress': proof_stress,
# # #             'elongation': elongation
# # #         })
# # #     except Exception as e:
# # #         logger.error(f"Prediction error: {str(e)}")
# # #         return jsonify({'error': str(e)}), 400

# # # if __name__ == '__main__':
# # #     app.run(debug=True)



# # from flask import Flask, request, render_template, jsonify
# # import pickle
# # import numpy as np
# # import logging

# # app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# # logging.basicConfig(level=logging.DEBUG)
# # logger = logging.getLogger(__name__)

# # try:
# #     with open('model.pkl', 'rb') as file:
# #         model = pickle.load(file)
# #     logger.info("Model loaded successfully")
# # except Exception as e:
# #     logger.error(f"Failed to load model: {str(e)}")
# #     raise

# # @app.route('/')
# # def home():
# #     return render_template('home.html')

# # @app.route('/aboutus')
# # def aboutus():
# #     return render_template('aboutus.html')

# # @app.route('/Applications')
# # def applications():
# #     return render_template('Applications.html')

# # @app.route('/blog')
# # def blog():
# #     return render_template('blog.html')


# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     """Predict the quality of the material based on input parameters."""


# #     logger.debug("Received predict request")
# #     try:
# #         tensile_strength = float(request.form['tensile_strength'])
# #         proof_stress = float(request.form['proof_stress'])
# #         elongation = float(request.form['elongation'])

# #         if tensile_strength <= 0 or proof_stress <= 0 or elongation <= 0:
# #             logger.warning("Invalid input: Negative or zero values")
# #             return jsonify({'error': 'All inputs must be positive numbers'}), 400

# #         input_data = np.array([[tensile_strength, proof_stress, elongation]])
# #         logger.debug(f"Input data: {input_data}")

# #         prediction = model.predict(input_data)[0]
# #         confidence = model.predict_proba(input_data).max() * 100
# #         logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")

# #         if prediction == 1:
# #             result = "Good for Industry Use"
# #             suggestion = "No improvements needed."        
# #         else:
# #             result = "Not Suitable for Industry Use"
# #             suggestion = "Increase annealing time and temperature to improve both tensile strength and ductility."

# #         return jsonify({
# #             'result': result,
# #             'confidence': f"{confidence:.2f}%",
# #             'suggestion': suggestion,
# #             'tensile_strength': tensile_strength,
# #             'proof_stress': proof_stress,
# #             'elongation': elongation
# #         })
# #     except Exception as e:
# #         logger.error(f"Prediction error: {str(e)}")
# #         return jsonify({'error': str(e)}), 400

# # if __name__ == '__main__':
# #     app.run(debug=True)


# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import pickle

# app = Flask(__name__, template_folder='templates', static_folder='static')

# # Load the model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Routes
# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/aboutus')
# def aboutus():
#     return render_template('aboutus.html')

# @app.route('/Application')
# def application():
#     return render_template('Application.html')

# @app.route('/product')
# def product():
#     return render_template('product.html')

# @app.route('/blog')
# def blog():
#     return render_template('blog.html')

# @app.route('/Prediction', methods=['GET', 'POST'])
# def prediction():
#     if request.method == 'POST':
#         try:
#             ts = float(request.form['tensile_strength'])
#             ps = float(request.form['proof_stress'])
#             el = float(request.form['elongation'])
#             input_data = np.array([[ts, ps, el]])
#             prediction = model.predict(input_data)[0]
#             confidence = model.predict_proba(input_data).max() * 100
#             result = "Good for Industry Use" if prediction == 1 else "Not Suitable for Industry Use"
#             suggestion = "No improvements needed." if prediction == 1 else "Consider increasing tensile strength or adjusting heat treatment."
#             return jsonify({
#                 'result': result,
#                 'confidence': f"{confidence:.2f}%",
#                 'suggestion': suggestion,
#                 'tensile_strength': ts,
#                 'proof_stress': ps,
#                 'elongation': el
#             })
#         except Exception as e:
#             return jsonify({'error': str(e)}), 400
#     return render_template('Prediction.html')




# from flask import Flask, request, render_template, jsonify
# import numpy as np
# import pickle

# app = Flask(__name__, template_folder='templates', static_folder='static')

# # Load the trained model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # ROUTES
# @app.route('/home')
# def home():
#     return render_template('home.html')

# @app.route('/aboutus')
# def aboutus():
#     return render_template('aboutus.html')

# @app.route('/Application')
# def application():
#     return render_template('Application.html')

# @app.route('/product')
# def product():
#     return render_template('product.html')

# @app.route('/blog')
# def blog():
#     return render_template('blog.html')

# @app.route('/Prediction', methods=['GET', 'POST'])
# def prediction():
#     if request.method == 'POST':
#         try:
#             ts = float(request.form['tensile_strength'])
#             ps = float(request.form['proof_stress'])
#             el = float(request.form['elongation'])

#             input_data = np.array([[ts, ps, el]])
#             prediction = model.predict(input_data)[0]
#             confidence = model.predict_proba(input_data).max() * 100

#             result = "Good for Industry Use" if prediction == 1 else "Not Suitable for Industry Use"
#             suggestion = "No improvements needed." if prediction == 1 else "Consider increasing tensile strength or adjusting heat treatment."

#             return jsonify({
#                 'success': True,
#                 'result': result,
#                 'confidence': f"{confidence:.2f}%",
#                 'suggestion': suggestion,
#                 'tensile_strength': ts,
#                 'proof_stress': ps,
#                 'elongation': el
#             })
#         except Exception as e:
#             return jsonify({'success': False, 'error': str(e)}), 500

#     return render_template('Prediction.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("❌ model.pkl not found. Please run train_model.py first.")
    model = None

# ROUTES
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/Application')
def application():
    return render_template('Application.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            if model is None:
                raise ValueError("Model not loaded. Please train the model first.")

            ts = float(request.form['tensile_strength'])
            ps = float(request.form['proof_stress'])
            el = float(request.form['elongation'])

            input_data = np.array([[ts, ps, el]])
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data).max() * 100

            result = "Good for Industry Use" if prediction == 1 else "Not Suitable for Industry Use"
            suggestion = "No improvements needed." if prediction == 1 else "Consider increasing tensile strength or adjusting heat treatment."

            return jsonify({
                'success': True,
                'result': result,
                'confidence': f"{confidence:.2f}%",
                'suggestion': suggestion,
                'tensile_strength': ts,
                'proof_stress': ps,
                'elongation': el
            })
        except Exception as e:
            print("❌ Error during prediction:", str(e))
            return jsonify({'success': False, 'error': str(e)}), 500

    return render_template('Prediction.html')

# Optional route to verify Flask is running
@app.route('/test')
def test():
    return "✅ Flask is running!"

if __name__ == '__main__':
    app.run(debug=True)
