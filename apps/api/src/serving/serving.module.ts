import { Module } from '@nestjs/common';
import { ServingController } from './serving.controller';
import { ServingService } from './serving.service';

@Module({
  controllers: [ServingController],
  providers: [ServingService]
})
export class ServingModule {}
