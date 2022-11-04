[![Build Status](https://travis-ci.org/chaosmail/tfjs-onnx.svg?branch=master)](https://travis-ci.org/chaosmail/tfjs-onnx)

# Tensorflow.js Onnx Runner

Run and finetune pretrained Onnx models in the browser with GPU support via the wonderful [Tensorflow.js][tfjs] library.

## Usage

### Installation

You can use this as standalone es5 bundle like this:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.10.3"></script>
<script src="https://unpkg.com/tfjs-onnx"></script>
```

Then, loading a model is as simple as referencing the path to the `model.onnx` file.

Here is an example of loading SqueezeNet:

```js
var modelUrl = 'models/squeezenet/model.onnx';

// Initialize the tf.model
var model = new onnx.loadModel(modelUrl);

// Now use tf.model
const pixels = tf.fromPixels(img);
const predictions = model.predict(pixels);
```

### Run Demos

To run the demo, use the following:

```bash
npm run build

# Start a webserver
npm run serve
```

Now navigate to http://localhost:8080/demos.

> Hint: some of the models are quite big (>30MB). You have to download the Onnx models and place them into the `demos/models` directory to save bandwith.

## Development

```sh
npm install
```

To build a standalone bundle run

```sh
npm run build
```

[tfjs]: https://github.com/tensorflow/tfjs
