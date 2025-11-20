import * as cdk from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import * as Backend from '../lib/backend-stack';

describe('NovaCdkStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const stack = new Backend.NovaCdkStack(app, 'TestStack');
    template = Template.fromStack(stack);
  });

  test('S3 Buckets Created', () => {
    template.resourceCountIs('AWS::S3::Bucket', 2);
  });

  test('Lambda Functions Created', () => {
    // CDK creates additional Lambda for custom resources (S3 auto-delete)
    template.resourceCountIs('AWS::Lambda::Function', 5);
  });

  test('Query Handler has Function URL', () => {
    template.hasResourceProperties('AWS::Lambda::Url', {
      AuthType: 'NONE',
    });
  });

  test('Bedrock Permissions Granted', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
          }),
        ]),
      },
    });
  });

  test('Vector Bucket has Encryption', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            ServerSideEncryptionByDefault: {
              SSEAlgorithm: 'AES256',
            },
          },
        ],
      },
    });
  });
});
