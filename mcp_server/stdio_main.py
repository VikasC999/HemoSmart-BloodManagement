"""
Runs the HemoSmart MCP server over stdio.

For a local MCP client (Claude Desktop, Claude Code, etc.) to connect,
point its config at this script, e.g. in Claude Desktop's
claude_desktop_config.json:

    {
      "mcpServers": {
        "hemosmart": {
          "command": "/absolute/path/to/HemoSmart-BloodManagement/.venv/bin/python",
          "args": ["-m", "mcp_server.stdio_main"],
          "cwd": "/absolute/path/to/HemoSmart-BloodManagement",
          "env": { "GROQ_API_KEY": "...", "HEMOSMART_LLM_PROVIDER": "groq" }
        }
      }
    }

The client launches this as a subprocess and talks to it over
stdin/stdout -- no network, no auth, which is why this path is for
local use. The HTTP-mounted server in backend/main.py is what a
deployed HemoSmart exposes remotely.
"""

from mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
