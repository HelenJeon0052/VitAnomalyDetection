import { Injectable, OnModuleDestroy } from '@nestjs/cli'
import * as ort from 'onnxruntime-node'


@Injectable()
export class SessionManager implements OnModuleDestroy {
    private readonly sessions = new Map<string, ort.InferenceSession>();

    async getOrCreate(modelPath: string): Promise<ort.InferenceSession> {
        const cached = this.sessions.get(modelPath)
        if (cached) return cached

        const session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'],
            graphOptimizationLevel: 'all',
        })

        this.sessions.set(modelPath, session)
        console.log(session)
        return session
    }

    getInputName(session: ort.InferenceSession): string {
        return session.inputNames[0]
    }

    getOutputName(session: ort.InferenceSession): string[] {
        return session.outputNames
    }

    async onModuleDestroy() {
        this.sessions.clear()
    }
}