
function getImageData(player, options) {
  options = options || {};
  options.shape = options.shape || null;
  options.gray = options.gray || false;
  options.crop = options.crop || false;

  return tf.tidy(() => {

    const img = tf.fromPixels(player);

    const croppedImg = options.crop ?
      cropImage(img) : img;

    const resizedImg = options.shape ?
      tf.image.resizeBilinear(croppedImg, options.shape) :
      croppedImg;

    const grayImg = options.gray ?
      rgbToGrayscale(resizedImg) :
      resizedImg;

    // Convert to float and add batch dimension
    return grayImg.cast('float32').expandDims();
  });
}

function displayImage(data, elemId) {
  const elem = document.getElementById(elemId);
  const pixels = tf.squeeze(data).div(tf.scalar(255));
  tf.toPixels(pixels, elem);
}

function displayLabel(probs, labels, elemId) {
  const elem = document.getElementById(elemId);
  const f = n => n.toLocaleString(undefined, { minimumFractionDigits: 2 });

  const row = labels.map((d,i) => {
    return labels[i] + ": " + f(probs[i] * 100) + "%";
  })

  elem.innerHTML = row.join("<br>");
}

// TODO port back to tfjs
// @src https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py#L1255
function rgbToGrayscale(input) {
  return tf.tidy(() => {
    const rgbWeights = tf.tensor1d([0.2989, 0.5870, 0.1140]);
    const floatInput = input.cast('float32');
    const grayImage = floatInput.mul(rgbWeights).sum(-1);
    return grayImage.expandDims(input.shape.length - 1);
  });
}

// TODO port back to tfjs
// @src https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/webcam.js#L56
function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, img.shape[2]]);
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);

  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  return [topkValues, topkIndices];
}
