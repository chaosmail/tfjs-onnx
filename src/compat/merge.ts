import * as tf from '@tensorflow/tfjs';
import {Shape, Tensor} from '@tensorflow/tfjs';
import {LayerConfig} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {Merge} from '@tensorflow/tfjs-layers/dist/layers/merge';
import * as generic_utils from '@tensorflow/tfjs-layers/dist/utils/generic_utils';

export class MergeLayer extends Merge {
  build(inputShape: Shape|Shape[]): void {
    // Used purely for shape validation.
    if (Array.isArray(inputShape) && !Array.isArray(inputShape[0])) {
      // Make sure that inputShape is an Array of shape.
      inputShape = [generic_utils.getExactlyOneShape(inputShape)];
    }
    inputShape = inputShape as Shape[];
    if (inputShape.length < 2) {
      throw new Error(
          'A merge layer should be called on an Array of at least 2 inputs.' +
          ` Got ${inputShape.length} input(s).`);
    }
  }

  computeOutputShape(inputShape: Shape|Shape[]): Shape|Shape[] {
    inputShape = [].concat(inputShape) as Shape[];
    return inputShape[0];
  }
}

export class AddLayer extends MergeLayer {
  static className = 'Add';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = tf.zeros(inputs[0].shape);
    for (const input of inputs) {
      output = tf.add(output, input);
    }
    return output;
  }
}

export class SubLayer extends MergeLayer {
  static className = 'Sub';
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

export class MulLayer extends MergeLayer {
  static className = 'Mul';
  constructor(config?: LayerConfig) {
    super(config as LayerConfig);
  }

  protected mergeFunction(inputs: Tensor[]): Tensor {
    let output = tf.ones(inputs[0].shape);
    for (const input of inputs) {
      output = tf.mul(output, input);
    }
    return output;
  }
}

export class DivLayer extends MergeLayer {
  static className = 'Div';
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
