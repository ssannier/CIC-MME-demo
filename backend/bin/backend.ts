#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib/core';
import { NovaCdkStack } from '../lib/backend-stack';

const app = new cdk.App();
new NovaCdkStack(app, 'NovaCdkStack', {
  env: { 
    account: process.env.CDK_DEFAULT_ACCOUNT, 
    region: process.env.CDK_DEFAULT_REGION || 'us-east-1' 
  },
});
