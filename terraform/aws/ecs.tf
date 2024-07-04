resource "aws_ecs_cluster" "ai_assistant_cluster" {
  name = "ai_assistant"
  configuration {
    execute_command_configuration {
      kms_key_id = aws_kms_key.key.arn
      logging    = "OVERRIDE"

      log_configuration {
        cloud_watch_encryption_enabled = true
        cloud_watch_log_group_name     = aws_cloudwatch_log_group.ai_assistant-cloudwatch-log.name
      }
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "cluster" {
  cluster_name = aws_ecs_cluster.ai_assistant_cluster.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
  }
}

resource "aws_cloudwatch_log_group" "ai_assistant-cloudwatch-log" {
  name = "/ecs/ai_assistant-taskdef-iac-https"
}

resource "aws_ecs_service" "ai_assistant_service" {
  name                 = "ai_assistant-service-iac-https"
  cluster              = aws_ecs_cluster.ai_assistant_cluster.id
  task_definition      = aws_ecs_task_definition.ai_assistant_task_definition.arn
  force_new_deployment = true
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    base              = 1
    weight            = 1
  }
  network_configuration {
    subnets          = [data.aws_subnet.ai_assistant_subnet_1.id, data.aws_subnet.ai_assistant_subnet_2.id, data.aws_subnet.ai_assistant_subnet_3.id]
    security_groups  = [aws_security_group.ai_assistant_security_group.id]
    assign_public_ip = true
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  desired_count = 1

  load_balancer {
    target_group_arn = aws_lb_target_group.ai_assistant_target_group_https.arn
    container_name   = "ai_assistant_https"
    container_port   = 80
  }

  depends_on = [aws_lb_listener.application_lb_listener]
}

resource "aws_ecs_task_definition" "ai_assistant_task_definition" {
  family                   = "ai_assistant-taskdef-iac-https"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]

  cpu    = "512"
  memory = "1024"
  volume {
    name = "efs-volume"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.ai_assistant_efs_file_system.id
      root_directory     = "/"
      transit_encryption = "ENABLED"
    }
  }
  volume {
    name = "configuration-secret"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.ai_assistant_efs_file_system.id
      root_directory     = "/"
      transit_encryption = "ENABLED"
    }
  }
  execution_role_arn = aws_iam_role.ai_assistant_ecs_execution_role.arn
  task_role_arn      = aws_iam_role.ai_assistant_ecs_execution_role.arn

  container_definitions = jsonencode([{
    name   = "ai_assistant_https"
    image  = var.ecr_image_url
    cpu    = 512
    memory = 1024
    runtime_platform = {
      "cpuArchitecture" : "X86_64",
      "operatingSystemFamily" : "LINUX"
    }
    mountPoints = [{
      sourceVolume  = "efs-volume"
      containerPath = "/app/data/chroma"
      readOnly      = false
    },
    {
      sourceVolume  = "configuration-secret"
      containerPath = "/app/.streamlit/secrets.toml"
      readOnly      = true
    }]
    memoryReservation = 1024
    portMappings = [{
      name          = "ai_assistant-80-tcp"
      containerPort = 80
      hostPort      = 80
      appProtocol   = "http"
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-create-group  = "true"
        awslogs-group         = "/ecs/ai_assistant-taskdef-iac"
        awslogs-region        = "eu-west-1"
        awslogs-stream-prefix = "ecs"
      },
    }
    essential = true
    environment = [
      {
        "name" : "LANGCHAIN_TRACING_V2",
        "value": "${var.langchain_tracing_v2_bool}"
      }
    ]
    secrets = [
      {
        "name" : "OPENAI_API_KEY",
        "valueFrom" : "${data.aws_secretsmanager_secret.secret.arn}:${var.openai_key_name}::"
      },
      {
        "name" : "MISTRAL_API_KEY",

        "valueFrom" : "${data.aws_secretsmanager_secret.secret.arn}:${var.mistral_key_name}::"
      },
      {
        "name" : "HF_TOKEN",

        "valueFrom" : "${data.aws_secretsmanager_secret.secret.arn}:${var.hf_token_name}::"
      },
      {
        "name" : "LANGCHAIN_API_KEY",

        "valueFrom" : "${data.aws_secretsmanager_secret.secret.arn}:${var.langchain_key_name}::"
      }
    ]
  }])
}
