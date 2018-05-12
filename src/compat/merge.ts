import * as tf from '@tensorflow/tfjs';
import {Tensor} from '@tensorflow/tfjs';
import {LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Merge} from '@tensorflow/tfjs-layers/dist/layers/merge';

// TODO port back to tfjs
export class SubCompat extends Merge {
  static className = 'SubCompat';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = tf.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = tf.sub(output, input);
    }
    return output;
  }
}

// TODO port back to tfjs
export class DivCompat extends Merge {
  static className = 'DivCompat';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = tf.ones(inputs[0].shape);
    for (const input of inputs) {
      output = tf.div(output, input);
    }
    return output;
  }
}
