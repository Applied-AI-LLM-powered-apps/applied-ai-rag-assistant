data "aws_iam_policy_document" "opensearch_domain_policy" {
  statement {
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = var.opensearch_allowed_users_and_policy_arn #list of arn
    }

    actions   = ["es:*"]
    resources = ["arn:aws:es:eu-west-1:441525731509:domain/ai-assistant/*"]
  }
  statement {
    principals {
      type        = "AWS"
      identifiers = var.opensearch_allowed_users_and_policy_arn #list of arn
    }

    actions   = ["es:*"]
    resources = ["arn:aws:es:eu-west-1:441525731509:domain/ai-assistant/*"]

    condition {
      test     = "IpAddress"
      variable = "aws:SourceIp"
      values   = ["172.16.1.0/8"]
    }
  }
}

resource "aws_opensearch_domain" "ai_assistant_opensearch_domain" {
  domain_name    = var.opensearch_domain_name
  engine_version = "OpenSearch_2.13"

  cluster_config {
    instance_type          = "r5.large.search"
    zone_awareness_enabled = false
    instance_count = 1
    multi_az_with_standby_enabled = false
  }

  advanced_security_options {
    enabled                        = true
    anonymous_auth_enabled         = false
    internal_user_database_enabled = true
    master_user_options {
      master_user_name     = var.opensearch_dashboard_user
      master_user_password = var.opensearch_dashboard_password
    }
  }
  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }

  node_to_node_encryption {
    enabled = true
  }

  ebs_options {
    ebs_enabled = true
    volume_size = 10
  }
  access_policies = data.aws_iam_policy_document.opensearch_domain_policy.json
}
