import { PreprocessProfile } from "./preprocess.profile";



export interface ModelProfile {
    id: string;
    modelPath: string;
    inputName: string;
    outputName: string;
    inputShape: [number, number, number, number];
    task: 'classification' | 'segmentation' | 'restruction';
    preprocess : PreprocessProfile;
}

export const MODEL_REGISTRY: Record<string, ModelProfile> = {
    vit_denoiser: {
        id: 'vit_denoiser',
        modelPath: './models/vit_denoiser.onnx',
        inputShape: [1, 1, 255, 255],
        task: 'classification',
        preprocess : {
            width:255,
            height:255,
            channels:8,
            colorMode:'grayscale',
            normalize: 'zero',
            layout: 'NCLW'
        },
    },
}