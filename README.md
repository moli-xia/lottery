# Lottery Prediction Assistant（魔力彩票助手）

基于历史数据抓取与大模型（LLM）分析的双色球（ssq）与大乐透（dlt）预测辅助系统，提供前台页面与后台管理（LLM 配置 / 手动抓取 / 测试）。

## 功能概览

- 自动抓取历史开奖数据，并按开奖时间智能定时更新
- 支持 OpenAI 兼容的 `/chat/completions` 接口（默认：siliconflow + DeepSeek-R1）
- 前端实时刷新（SSE），后端 FastAPI + SQLite 持久化
- 后台 `/admin`：配置 LLM、测试连通性、修改后台密码

## Docker 部署（推荐）

镜像地址：`superneed/lottery`（支持 `linux/amd64` 与 `linux/arm64`）

### 1) 从 Docker Hub 拉取镜像

```bash
docker pull superneed/lottery:latest
```

### 2) 运行容器（带数据持久化）

```bash
touch lottery.db

docker run -d \
  --name lottery \
  -p 8888:8888 \
  -v "$(pwd)/lottery.db:/app/lottery.db" \
  -e TZ=Asia/Shanghai \
  -e ADMIN_USERNAME=admin \
  -e ADMIN_PASSWORD=admin \
  -e SESSION_SECRET="$(openssl rand -hex 32)" \
  --restart always \
  superneed/lottery:latest
```

- 前台：`http://<your-ip>:8888/`
- 后台：`http://<your-ip>:8888/admin`（默认账号密码：`admin/admin`，建议首次登录后修改密码）

### 3) 在后台配置 LLM

进入 `/admin` 后可配置：

- LLM API Key（必填）
- LLM Base URL（可选，默认 `https://api.siliconflow.cn/v1`）
- LLM 模型名（默认 `deepseek-ai/DeepSeek-R1`）

### 常用运维命令

```bash
docker logs -f lottery
docker restart lottery
docker exec -it lottery python -V
```

### 升级镜像（不丢数据）

```bash
docker pull superneed/lottery:latest
docker rm -f lottery
docker run -d \
  --name lottery \
  -p 8888:8888 \
  -v "$(pwd)/lottery.db:/app/lottery.db" \
  -e TZ=Asia/Shanghai \
  -e ADMIN_USERNAME=admin \
  -e ADMIN_PASSWORD=admin \
  -e SESSION_SECRET="$(openssl rand -hex 32)" \
  --restart always \
  superneed/lottery:latest
```

## 从源码构建 Docker 镜像

### 单机构建（本地使用）

```bash
docker build -t superneed/lottery:local .
docker run --rm -p 8888:8888 superneed/lottery:local
```

### 多架构构建并推送到 Docker Hub

仓库内已提供脚本 [build_docker.sh](file:///www/wwwroot/gamble/build_docker.sh)：

```bash
docker login
bash build_docker.sh
```

等价手动命令如下：

```bash
docker login
docker buildx create --use --name mybuilder || docker buildx use mybuilder
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t superneed/lottery:latest \
  --push \
  .
```

## 本地开发

### 环境要求

- Python 3.9+
- Node.js 18+

### 后端启动

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8888
```

### 前端启动

```bash
cd frontend
npm install
npm run dev
```

默认前端开发服务器由 Vite 提供，后端 API 运行在 `http://localhost:8888`。

## 配置项（环境变量）

- `TZ`：时区（默认 `Asia/Shanghai`）
- `ADMIN_USERNAME`：后台用户名（默认 `admin`）
- `ADMIN_PASSWORD`：后台初始密码（默认 `admin`，首次修改密码后以数据库内保存为准）
- `SESSION_SECRET`：后台会话密钥（默认 `dev-secret`，生产环境务必设置为随机值）

## 目录结构

```
.
├── backend/            # FastAPI 后端
├── frontend/           # Vue 3 前端
├── Dockerfile          # 多阶段构建：前端构建 + 后端运行
└── build_docker.sh     # buildx 多架构构建并推送
```

## License

MIT
