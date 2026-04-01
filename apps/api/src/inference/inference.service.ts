import { Injectable, NotFoundException } from '@nestjs/common';
import { SessionManager } from '../serving/session.manager';


import { PreprocessService } from '../serving/preprocessing.service';
import { PostprocessService } from '../serving/postprocess.service';
import { InputSpecValidator } from '../validators/input-spec.validator';

import  { MODEL_REGISTRY } from '../serving/profiles/model.profile';


@Injectable()
export class InferenceService {
  constructor(
    private readonly sessionManager: SessionManager,
    private readonly preprocessService: PreprocessService,
    private readonly postprocessService: PostprocessService,
    private readonly inputSpecValidator: InputSpecValidator,
  ) {}

  async run(modelId: string, buffer: Buffer) {
    const profile = MODEL_REGISTRY[modelId]
    if (!profile) {
      throw new NotFoundException(`Model not found: ${modelId}`)
    }

    const session = await this.sessionManager.getOrCreate(profile.modelPath);
    const inputName = profile.inputName ?? session.inputNames[0]

    const tensor = await this.preprocessService.toTensor(buffer, profile.preprocess)
    this.inputSpecValidator.validteTensor(tensor, profile)

    const startedAt = Date.now()
    const outputs = await session.run({ [inputName]: tensor })
    const latencyMs = Date.now() - startedAt

    const result = this.postprocessService(outputs, profile)

    return {
      ok: true, modelId, latencyMs, result,
    }
  }
}