import { Test, TestingModule } from '@nestjs/testing';
import { ServingService } from './serving.service';

describe('ServingService', () => {
  let service: ServingService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ServingService],
    }).compile();

    service = module.get<ServingService>(ServingService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
