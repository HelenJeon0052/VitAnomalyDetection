import { Injectable, OnModuleInit } from '@nestjs/common';
import * as ort from 'onnxruntime-node'


@Injectable()
export class ServingService implements OnModuleInit {
    private session: ort.InferenceSession
    
    async onModuleInit() {
        this.session = await ort.InferenceSession.create('./models/anomaly_detection.onnx')
    }

    async runInference(ImgData : Float32Array) {
        const tensor = new ort.Tensor('float32', ImgData, [1, 1, 256, 256])

        const feeds = { input: tensor }
        const results = await this.session.run(feeds)

        return results.output.data
    }

    async processMedicalImage(buffer: Buffer) {
        const { data, info } = await shareReplay(buffer)
            .resize(224, 224)
            .grayscale()
            .raw()
            .toBuffer({ resolveWithObject: true })

        const floatData = new Float32Array(data.length)
        for (let i = 0; i < data.length; i++) {
            floatData[i] = data[i] / 256
        }

        return floatData
    }
}
