# python-fastapi-postgresql-template

使用方法

把这些文件放到你的项目里

打开 VS Code → Ctrl+Shift+P → Dev Containers: Reopen in Container

等待容器启动

FastAPI 应用跑在 http://localhost:8000

PostgreSQL 数据库在 localhost:5432

pgAdmin 在 http://localhost:5050（账号 admin@admin.com / 密码 admin）

你就有了一个完整的 FastAPI + 数据库 + 管理工具 的开发环境！ 🎉

# ###########################################################################################
初始化 Alembic

在容器里执行：

alembic init alembic


会生成一个 alembic/ 目录和 alembic.ini 文件。

修改 alembic/env.py，让它识别 SQLAlchemy 的 Base：

from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from database import Base, DATABASE_URL
import models  # 确保 Alembic 知道模型

# Alembic 配置
config = context.config
fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(
        url=DATABASE_URL, target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        {"sqlalchemy.url": DATABASE_URL}, prefix="", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

6. 使用 Alembic 管理迁移

生成迁移脚本：

alembic revision --autogenerate -m "create users table"


应用迁移（升级数据库）：

alembic upgrade head


如果表结构有变化（比如加字段），只要修改 models.py，再运行：

alembic revision --autogenerate -m "add new field"
alembic upgrade head


👉 Alembic 会帮你生成 SQL 脚本，数据库结构会自动更新。