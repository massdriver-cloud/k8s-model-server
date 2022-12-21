import os
import uuid
import threading

from diffusers import StableDiffusionPipeline
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
sem = threading.Semaphore()

@app.route("/", methods=['POST', 'PUT'])
def infer():
    body = request.json
    phrase = body.get("phrase", "flamingos")
    model  = body.get("model", "runwayml/stable-diffusion-v1-5")
    steps = int(body.get("steps", 50))
    if steps > 500:
        return jsonify({"error": "Steps must be less than 500"})
    guidance_scale  = float(body.get("guidance_scale", 8.5))
    sem.acquire()
    print(f"phrase: {phrase}\nmodel: {model}\nsteps: {steps}\nguidance_scale: {guidance_scale}")
    pipe = StableDiffusionPipeline.from_pretrained(
      model,
      use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    ).to("cuda")
    result = pipe(phrase, num_inference_steps=steps)
    image = result.images[0]
    image_path = f"/tmp/stable-{uuid.uuid4()}.png"
    image.save(image_path)
    sem.release()
    return send_file(image_path, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
