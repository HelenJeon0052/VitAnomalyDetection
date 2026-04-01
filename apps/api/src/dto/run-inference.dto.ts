import { IsString } from 'class-validator'




export class RunInferenceDto {
  @IsString()
  modelId: string
}