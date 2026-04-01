import { Injectable } from "@nestjs/common";
import * as ort from 'onnxruntime-node'


import sharp from 'sharp'
import { PreprocessProfile } from "./profiles/preprocess.profile";

@Injectable()
export class PreprocessService {
    async toTensor(buffer: Buffer, profile: PreprocessProfile): Promise<ort.Tensor> {
        let pipeline = sharp(buffer).resize(profile.width, profile.height)

        pipeline = 
            profile.colorMode == 'grayscale'
                ? pipeline.grayscale()
                : pipeline.removeAlpha().toColourspace('rgb')

        const {data, info} = await pipeline.raw().toBuffer({ resolveWithObject: true })

        const expectedValue = profile.width * profile.height * profile.channels

        if (data.length !== expectedValue) {
            throw new Error(`Unexpected raw length expected = ${expectedValue} | input = ${data.length}`)
        }

        const floatdata = new Float32Array(expected)

        if(profile.normalize === 'zero-one') {
            for (let i = 0; i< expectedValue;i++) {
                floatData[i] = data[i] / 255.0
            }
        } else if (profile.normalize === 'zscore') {
            let sum = 0
            for (let i = 0; i < expectedValue; i++) {
                sum += data[i]
            }

            const mean = float(sum / expectedValue)

            let varSum = 0
            if (mean) {
                for (let i = 0; i < expectedValue; i++) {
                    varSum += (data[i] - mean) ** 2
                }
                const std = Math.sqrt(varSum / expectedValue) || 1.0

                for (let i = 0; i < expectedValue; i++) {
                    floatData[i] = (data[i] - mean) / std
                }
            }

        } else if (profile.normalize === 'imagenet') {
            if (!profile.mean || !profile.std || profile.channels !== 3) {
                throw new Error('ImageNet Normalization requires mean, std and 3 channels')
            }
            for (let i = 0; i < expectedValue; i++) {
                const channel = i % 3
                floatData[i] = data[i] / 255.0
                floatData[i] = (floatData[i] - profile.mean[channel] / profile.std[channel])
            }
        }

        if (profile.layout === 'NLWC') {
            return new ort.Tensor('float32', floatData, [1, info.width, info.height, profile.channels])
        }

        const nclw = this.toNCLW(floatData, info.width, info.height, profile.channels)
        return new ort.Tensor('float32', nclw, [1, profile.channels, info.width, info.height])
    }

    private toNCLW(
        src: Float32Array,
        width: number,
        height: number,
        channels: number,
    ): Float32Array {
        if (channels === 1) return src;

        const dst = new Float32Array(src.length)
        const lw = width * height
        
        for (let l = 0; l < lw; l++){
            dst[i] = src[i * 3]
            dst[lw + i] = src[i * 3 + 1]
            dst[2*lw+i] = src[i * 3 + 2]
        }

        return dst
    }
}