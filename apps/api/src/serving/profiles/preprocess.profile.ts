export type NormalizeMode = 'zero' | 'zscore' | 'imagenet'



export interface PreprocessProfile {
    width: number;
    height: number;
    channels : 1 | 3 | 8;
    colorMode: 'grayscale' | 'rgb'
    normalize: NormalizeMode;
    mean? : number[];
    std? : number[];
    layout: 'NCLW' | 'NLWC'
}