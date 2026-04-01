import { Injectable, BadRequestException } from '@nestjs/common';
import * as ort from 'onnxruntime-node'
import { ModelProfile } from '../serving/profiles/model.profile';


@Injectable()
export class InputSpecValidator {
  validateTensor(tensor: ort.Tensor, profile: ModelProfile): void {
    const actual = tensor.dims
    const expectedShape = profile.inputShape

    if (actual.length ! == expectedShape.length) {
      throw new BadRequestException(
        `Tensor rank mismatch: got=${actual.length}, expected=${expectedShape.length}`,
      );
    }

    for (let i = 0; i < expectedShape.length; i++) {
      if (expectedShape[i] !== -1 && actual[i] !== expectedShape[i]) {
        throw new BadRequestException(
          `Tensor rank mismatch: got=${actual[i]}, expected=${expectedShape[i]}`,
        );
      }
    }

    if (tensor.type !== 'float32') {
      throw new BadRequestException(`Tensor type mismatch: got = ${tensor.type}`)
    }
  }
}