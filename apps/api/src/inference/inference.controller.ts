import { Controller, BadRequestException, Body, Controller, Post, UploadedFile, UseInterceptors } from '@nestjs/common';



import { FileInterceptor } from '@nestjs/platform-express';
import { InferenceService } from './inference.service';
import { RunInferenceDto } from '../dto/run-inference.dto';

@Controller('inference')
export class InferenceController {
  constructor(private readonly inferenceService: InferenceService) {}

  @Post()
  @UseInterceptors(FileInterceptor('file'))
  async run(
    @UploadedFile() file: Express.Multer.File,
    @Body() dto: RunInferenceDto,
  ) {
    if (!file) {
      throw new BadRequestException('file must be provided')
    }

    return this.inferenceService.run(dto.modelId, file.buffer)
  }
}