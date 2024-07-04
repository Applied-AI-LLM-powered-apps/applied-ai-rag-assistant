variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}

variable "subnet_id_1" {
  description = "The ID of the first subnet"
  type        = string
}

variable "subnet_id_2" {
  description = "The ID of the second subnet"
  type        = string
}

variable "subnet_id_3" {
  description = "The ID of the third subnet"
  type        = string
}

variable "dns_url" {
  description = "The base DNS url (without subnet)."
  type        = string
}

variable "dns_url_app_subnet" {
  description = "The DNS URL of your application. (It need to have a valid HTTPS certificate and a route 53 hosted zone)"
  type        = string
}

variable "ecr_image_url" {
  description = "The AI assistant demonstrator ECR URL"
  type        = string
}

variable "secret_name" {
  description = "SecretManager secret name"
  type        = string
}

variable "openai_key_name" {
  description = "OpenAI key name in SecretManager secret"
  type        = string
}

variable "mistral_key_name" {
  description = "Mistral key name in SecretManager secret"
  type        = string
}


variable "hf_token_name" {
  description = "HF token name in SecretManager secret"
  type        = string
}

variable "langchain_key_name" {
  description = "Langchain key name in SecretManager secret"
  type        = string
}

variable "langchain_tracing_v2_bool" {
  description = "Langchain tracing V2 boolean string ('true' or 'false')"
  type        = string
}

variable "opensearch_domain_name" {
  description = "Name of the Opensearch domain"
  type        = string
}

variable "opensearch_dashboard_user" {
  description = "Username for Opensearch Dahsboard user"
  type        = string
}

variable "opensearch_dashboard_password" {
  description = "Password for Opensearch Dahsboard user"
  type        = string
}

variable "opensearch_allowed_users_and_policy_arn" {
  description = "List of allowed users and roles"
  type        = list(string)
}

