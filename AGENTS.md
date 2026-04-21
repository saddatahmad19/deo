<claude-mem-context>
# Memory Context

# [deo] recent context, 2026-04-20 10:21pm CDT

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 28 obs (10,218t read) | 0t work

### Apr 20, 2026
1142 9:45p 🔵 Azure OpenAI Connection Architecture in deo Agent Project
1143 9:46p 🔵 Malformed AZURE_OPENAI_ENDPOINT Causes "Resource Doesn't Exist" Error
1144 " 🔵 deo Agent Config System: TOML-Based Hierarchical Model Provider Selection
1145 9:47p 🔵 Three Azure Config Issues: No .deo.toml, Wrong Endpoint Domain, load_dotenv Missing in CLI
1146 " 🔵 load_dotenv Ordering Is Safe; ForgeConfig Name Bug Found in CLI
1147 9:49p 🔵 LangChain Expects OPENAI_API_VERSION Env Var, Not AZURE_OPENAI_API_VERSION
1148 " 🔵 AzureChatOpenAI Constructor Parameter Names: api_key vs openai_api_key
1149 9:51p 🔵 Microsoft Docs Confirm Correct Azure OpenAI Endpoint Domain and api_version Parameter
1150 " 🔵 Two Valid Azure OpenAI Base URL Domains; cognitiveservices.azure.com Is Not Valid
1151 9:53p 🔵 deo Project Uses uv venv at .venv/bin/python; python Command Not in PATH
1152 " 🔵 Confirmed: Malformed Endpoint Creates Broken URL with Query String in Path
1153 9:54p 🔵 Correct AZURE_OPENAI_ENDPOINT Value Confirmed: Must Be Bare Base URL with openai.azure.com Domain
1154 9:56p 🔴 Fixed Azure Endpoint Normalization in deo/agents/factory.py
1155 9:58p ✅ Updated .env.example with Azure Endpoint Format Warning and Valid Examples
1156 10:00p 🔵 Endpoint Normalization Fix Confirmed Working; cognitiveservices Domain Remains in .env
1157 " 🔵 api-version Correctly Extracted from Endpoint Query String as 2025-04-01-preview
1158 10:01p 🔵 ChromaDB Cloud Connection Blocks Agent Startup After Azure Fix</thinking>
1159 10:02p 🔵 ChromaDB Fails Even Without Cloud Credentials — Falls Back to Local Server Attempt
1160 " 🔵 ChromaDB Error Persists with Empty-String Credentials — CloudClient Triggers on Empty Strings
1161 10:03p 🔵 ChromaManager is Hardcoded to CloudClient Only — No Local Fallback Exists
1162 " ✅ Added OPENAI_API_VERSION as api_version Fallback in factory.py Azure Branch
1163 10:08p 🔵 Factory.py Changes Are Already Committed to HEAD — Git Working Tree Is Clean
1164 10:09p ✅ Updated .env: Stripped Endpoint Path and Fixed API Version Suffix
1165 10:13p 🔴 Azure OpenAI .env endpoint and API version corrected
1166 " 🔵 Azure OpenAI SDK URL construction confirmed post-fix
1167 10:14p 🔵 Azure endpoint DNS resolution fails — resource hostname unreachable
1168 10:16p 🔵 Azure OpenAI endpoint reachable — API enforces max_output_tokens minimum of 16
1169 10:17p 🔴 Azure OpenAI connection fully verified end-to-end for deo project
</claude-mem-context>