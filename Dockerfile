# 售后工单 Agent 应用镜像（Node 20 ESM，纯 JS 栈，无需编译工具链）
FROM node:20-bookworm-slim

ENV NODE_ENV=production \
    PORT=3204 \
    NPM_CONFIG_LOGLEVEL=warn

WORKDIR /app

# 先拷依赖清单，利用 Docker 层缓存；npm 走国内镜像加速
COPY package.json package-lock.json ./
RUN npm config set registry https://registry.npmmirror.com \
    && npm ci --omit=dev

# 拷贝源码与只读数据（data/docs 种子文档、orders.json、eval-cases.json）
COPY . .

# 非 root 运行
USER node

EXPOSE 3204

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||3204)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["npm", "start"]
