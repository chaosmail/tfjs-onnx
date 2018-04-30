[![Build Status](https://travis-ci.org/chaosmail/tfjs-onnx.svg?branch=master)](https://travis-ci.org/chaosmail/tfjs-onnx)

# Tensorflow.js Onnx Runner

Run pretrained Onnx models in the browser with GPU support via the wonderful [Tensorflow.js][tfjs] library.

## Usage

### Installation

You can use this as standalone es5 bundle like this:

```html
<script src="https://unpkg.com/tfjs-onnx"></script>
```

Then loading model is a simple as referencing the path to the caffemodel and prototxt files.

Here is an example of loading GoogLeNet:

```js
var modelUrl = 'models/bvlc_googlenet/model.onnx';

// Initialize the Onnx model
var model = new tf.OnnxModel(modelUrl);
```

This is how you load Squeezenet directly from Github:

```js
// The model is served entirely from Github
var GITHUB_CDN = 'https://rawgit.com/';

var modelUrl = GITHUB_CDN + 'DeepScale/SqueezeNet/master/SqueezeNet_v1.1/model.onnx';

// Initialize the Onnx model
var model = new tf.OnnxModel(modelUrl);
```

### Run Demos

To run the demo, use the following:

```bash
npm run build

# Start a webserver
npm run start
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
