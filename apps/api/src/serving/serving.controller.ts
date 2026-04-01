import { Controller, Post, UseInterceptors, UploadedFile, BadRequestException } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express'
import { ServingService } from './serving.service';



@Controller('serving')
export class ServingController {
    constructor(private readonly servingService: ServingService) {}

    @Post('analyze')
    @UseInterceptors(FileInterceptor('image'))
    async uploadImage(@UploadedFile() file: Express.Multer.File) {
        if(!file) {
            throw new BadRequestException('no image found')
        }

        const result = await this.servingService.processMedicalImage(file.buffer)
        return { success: true, anomalyScore: result }
    }
}
