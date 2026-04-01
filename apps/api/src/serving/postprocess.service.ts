import { Injectable } from '@nestjs/common'
import * as ort from 'onnxruntime-node'
import { ModelProfile } from './profiles/model.profile'



@Injectable()
export class PostprocessService {
    process (
        outputs: Record<string, ort.Tensor>,
        profile: ModelProfile
    ) : Record<string, unknown> {
        if (profile.task === 'classification') {
            return this.processClassification(outputs)
        }

        if (profile.task === 'segmentation') {
            return this.processSegmentation(outputs)
        }

        if (profile.task === 'reconstruction') {
            return this.processReconstruction(outputs)
        }

        return { raw: outputs }
    }

    private processClassification(outputs: Record<string, ort.Tensor>) {
        const first = Object.values(outputs)[0]
        const logits = Array.from(first.data as Float32Array)

        const probs = this.softmax(logits)
        const maxProb = Math.max(...probs)
        const pred = probs.indexOf(maxProb)

        return {
            logits,
            probailities : probs,
            predictedClass : pred,
            confidence: maxProb,
        }
    }

    private processSegmentation (outputs: Record<string, ort.Tensor>) {
        const first = Object.values(outputs)[0]
        return {
            shape: first.dims,
            mask: Array.from(this.data as Float32Array)
        }
    }

    private processReconstruction(outputs: Record<string, ort.Tensor>) {
        const first = Object.values(outputs)[0]
        return {
            shape: first.dims,
            reconstruction: Array.from(first.data as Float32Array)
        }
    }

    private softmax (logits: number[]): number[] {
        const maxLogit = Math.max(...logits)
        const exps = logits.map((x) => Math.exp(x - maxLogit))
        const sum = exps.reduce((a, b) => a + b, 0)
        return exps.map((x) => x / sum)
    }
}