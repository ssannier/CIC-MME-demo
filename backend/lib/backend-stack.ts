import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as iam from 'aws-cdk-lib/aws-iam';
import { RemovalPolicy } from 'aws-cdk-lib';
import path from 'path';

export class NovaCdkStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);



    // S3 Vector Bucket to hold vector embeddings
    const vectorBucket = new s3.Bucket(this, 'VectorBucket', {
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    // Lambda to handle queries with Nova MME
    const queryHandlerFn = new lambda.Function(this, 'QueryHandlerFunction', {
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda', 'query-handler')),
      timeout: cdk.Duration.seconds(30),
      memorySize: 512,
      environment: {
        VECTOR_BUCKET: vectorBucket.bucketName,
        BEDROCK_MODEL_ID: 'amazon.nova-lite-v1:0',
        EMBEDDING_MODEL_ID: 'amazon.titan-embed-text-v2:0',
      },
    });

    // Add function URL for query handler (public endpoint for frontend)
    const queryHandlerUrl = queryHandlerFn.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      cors: {
        allowedOrigins: ['*'],
        allowedMethods: [lambda.HttpMethod.ALL],
        allowedHeaders: ['*'],
        maxAge: cdk.Duration.hours(1),
      },
    });

    // Grant S3 read/write permissions to query handler
    vectorBucket.grantReadWrite(queryHandlerFn);

    // Bedrock permissions for embeddings and inference
    queryHandlerFn.addToRolePolicy(
      new iam.PolicyStatement({
        actions: [
          'bedrock:InvokeModel',
          'bedrock:InvokeModelWithResponseStream',
        ],
        resources: ['*'],
      })
    );

    // Amplify App using L1 construct
    const amplifyApp = new cdk.CfnResource(this, 'AmplifyApp', {
      type: 'AWS::Amplify::App',
      properties: {
        Name: 'NovaChatbotFrontend',
        Repository: 'https://github.com/ssannier/CIC-MME-demo',
        AccessToken: cdk.SecretValue.secretsManager('github-token').unsafeUnwrap(),
        BuildSpec: `version: 1
frontend:
  phases:
    preBuild:
      commands:
        - cd frontend
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: frontend/.next
    files:
      - '**/*'
  cache:
    paths:
      - frontend/node_modules/**/*`,
        EnvironmentVariables: [
          {
            Name: 'NEXT_PUBLIC_QUERY_URL',
            Value: queryHandlerUrl.url,
          },
        ],
      },
    });

    // Amplify Branch
    const amplifyBranch = new cdk.CfnResource(this, 'AmplifyBranch', {
      type: 'AWS::Amplify::Branch',
      properties: {
        AppId: amplifyApp.ref,
        BranchName: 'frontend',
        EnableAutoBuild: true,
      },
    });

    // Outputs
    new cdk.CfnOutput(this, 'VectorBucketName', {
      value: vectorBucket.bucketName,
      description: 'S3 bucket for vector embeddings',
    });

    new cdk.CfnOutput(this, 'QueryHandlerUrl', {
      value: queryHandlerUrl.url,
      description: 'Lambda Function URL - use this in your frontend',
    });

    new cdk.CfnOutput(this, 'QueryHandlerFunctionName', {
      value: queryHandlerFn.functionName,
      description: 'Lambda function name',
    });

    new cdk.CfnOutput(this, 'AmplifyAppUrl', {
      value: `https://${amplifyBranch.getAtt('BranchName')}.${amplifyApp.getAtt('DefaultDomain')}`,
      description: 'Amplify hosted frontend URL',
    });
  }
}
