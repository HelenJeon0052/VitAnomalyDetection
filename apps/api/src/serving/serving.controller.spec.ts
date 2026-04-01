import { Test, TestingModule } from '@nestjs/testing';
import { ServingController } from './serving.controller';

describe('ServingController', () => {
  let controller: ServingController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [ServingController],
    }).compile();

    controller = module.get<ServingController>(ServingController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
