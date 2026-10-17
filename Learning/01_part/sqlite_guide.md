# SQLite 详细介绍与使用指南

SQLite 是世界上部署最广泛的、开源的嵌入式关系型数据库引擎。它以**轻量级、零配置、高可靠性**著称，也是许多本地向量数据库（如 Chroma）、流式系统、客户端应用以及移动 App（iOS / Android）的底层存储底座。

---

## 一、 SQLite 的核心特点

### 1. 无服务器 (Serverless)
与 MySQL、PostgreSQL 等传统的客户端-服务器（Client-Server）架构不同，SQLite **不需要在后台运行一个独立的数据库进程**。
* 数据库引擎作为库（Library）直接链接并运行在你的应用程序进程中。
* 应用程序通过函数调用直接读写磁盘上的数据库文件，免去了网络 I/O 开销。

### 2. 单一磁盘文件 (Single Database File)
一个完整的 SQLite 数据库（包括表定义、索引、触发器以及所有数据）均被存储在**磁盘上的一个单独文件中**。
* 易于拷贝、备份和共享（只需要复制一个文件）。
* 支持内存数据库模式（通过传入路径 `:memory:`），在内存中运行，速度极快，进程结束即销毁。

### 3. 零配置 (Zero-Configuration)
* 不需要安装、不需要配置服务端口、不需要初始化用户权限与角色。
* 只要指定一个文件路径，SQLite 会在文件不存在时自动创建它并准备就绪。

### 4. 独特的动态弱类型系统 (Manifest Typing)
* 在大多数传统关系型数据库中，列（Column）的数据类型是固定的。
* SQLite 中，数据类型是**关联到数据值本身**，而不是列。即便你把一列定义为 `INTEGER`，你依然可以在其中插入 `VARCHAR`，SQLite 不会报错（除了声明为 `INTEGER PRIMARY KEY` 的列必须是整型）。
* 它主要支持 5 种存储类：`NULL`, `INTEGER`, `REAL` (浮点数), `TEXT`, `BLOB` (二进制)。

### 5. 严格的 ACID 事务支持
* 尽管它轻量，但它全面支持事务，确保了原子性 (Atomicity)、一致性 (Consistency)、隔离性 (Isolation) 和持久性 (Durability)。
* 支持回滚，即便遇到进程崩溃、死机或断电，数据也不会损坏。

---

## 二、 SQLite 的并发读写与 WAL 模式

SQLite 默认采用**读写锁**机制来控制并发：
* **多读单写**：允许多个进程/线程同时读取数据，但在写入数据时，会锁定整个数据库文件，其他所有读写操作都会被阻塞或返回 `database is locked`。

### WAL (Write-Ahead Logging) 模式
为了提升并发性能，SQLite 引入了 WAL 模式。
* **工作原理**：写入操作不直接修改原始数据库文件，而是写入一个附加的 `.wal` 日志文件中。在特定时机（Checkpoint），WAL 文件中的修改会被合并回主数据库文件。
* **优势**：
  * **读写不互斥**：在写入发生时，读取操作可以直接读取原始文件，从而实现“读写并发”。
  * **磁盘 I/O 效率更高**：多数写入变成顺序写入。
* **开启 WAL 方式**：
  ```sql
  PRAGMA journal_mode=WAL;
  ```

---

## 三、 Python 中使用 SQLite 实战

Python 自带了对 SQLite 的支持，无需额外安装任何包，直接导入 `sqlite3` 即可。

### 1. 基础 CRUD 操作示例

```python
import sqlite3

# 1. 连接到数据库（如果文件不存在，会自动创建）
# 也可以使用 ':memory:' 创建纯内存数据库
conn = sqlite3.connect("my_database.db")

# 2. 创建游标对象，用于执行 SQL 语句
cursor = conn.cursor()

# 3. 创建表
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    email TEXT UNIQUE
)
""")

# 4. 插入数据 (推荐使用占位符 ? 以防止 SQL 注入)
try:
    cursor.execute(
        "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
        ("张三", 25, "zhangsan@example.com")
    )
    
    # 批量插入示例
    users_list = [
        ("李四", 30, "lisi@example.com"),
        ("王五", 22, "wangwu@example.com")
    ]
    cursor.executemany(
        "INSERT INTO users (name, age, email) VALUES (?, ?, ?)",
        users_list
    )
    
    # 只有在 commit 之后，修改才会真正持久化到磁盘上
    conn.commit()
    print("数据插入成功！")
except sqlite3.IntegrityError as e:
    # 遇到唯一键冲突等异常时回滚事务
    conn.rollback()
    print(f"数据插入失败，已回滚事务。错误: {e}")

# 5. 查询数据
cursor.execute("SELECT * FROM users WHERE age > ?", (20,))
rows = cursor.fetchall()
for row in rows:
    # row 是一个元组，例如 (1, '张三', 25, 'zhangsan@example.com')
    print(f"ID: {row[0]}, 姓名: {row[1]}, 年龄: {row[2]}, 邮箱: {row[3]}")

# 6. 关闭游标与连接（记得释放资源）
cursor.close()
conn.close()
```

### 2. 使用 Context Manager（上下文管理器）自动打理事务
Python 的 `sqlite3.connect` 支持上下文管理器，它可以自动在无异常时 `commit`，有异常时 `rollback`：

```python
import sqlite3

with sqlite3.connect("my_database.db") as conn:
    cursor = conn.cursor()
    # 自动开启事务并在退出 with 块时自动 commit
    cursor.execute("UPDATE users SET age = age + 1 WHERE name = ?", ("张三",))
    # 注意：连接关闭仍需手动或在最外层处理
```

---

## 四、 SQLite 性能优化技巧

如果你在生产或高负载环境中使用 SQLite，以下优化手段可以带来数倍的性能飞跃：

1. **合并单条写为批量写（最重要）**：
   * 默认情况下，SQLite 每执行一条写 SQL 都会隐式开启并提交一个事务。这意味着每次写入都要进行一次磁盘同步（fsync），速度极慢（每秒只能写入几十条）。
   * 优化：显式地将大量写入放在**同一个 `BEGIN TRANSACTION` 和 `COMMIT`** 之间（或使用 `executemany`），每秒写入速度可以从几十条飙升到数万条。

2. **开启 WAL 模式**：
   * 可以显著降低 `database is locked` 报错的发生频率，提高读写吞吐量。
   * 可以在连接后执行 `conn.execute("PRAGMA journal_mode=WAL;")`。

3. **合理配置同步模式 (Synchronous)**：
   * 默认的 `PRAGMA synchronous=FULL;` 会在关键步骤强制刷盘，以提供最高的数据安全性。
   * 如果可以容忍极其罕见的操作系统崩溃时丢失少量事务（数据库文件不会损坏），可以设置为 `PRAGMA synchronous=NORMAL;`。在 WAL 模式下，设置为 `NORMAL` 是推荐的选择。

4. **调大 Cache 大小**：
   * `PRAGMA cache_size = -2000;`（负数表示以 KB 为单位，此处为 2MB 缓存空间；正数表示页的数量）。增大缓存可以减少磁盘的重复读取。

---

## 五、 SQLite 的局限性：何时不应该使用？

虽然 SQLite 极其优秀，但它有以下天然的硬伤，不适用于以下场景：

| 局限性维度 | 具体表现 | 解决方案 |
| :--- | :--- | :--- |
| **高并发写入** | 即使开启 WAL 模式，SQLite 也**只支持单写操作**。如果有几十个线程同时争抢写入，必会引发写入排队和超时。 | 换用 **PostgreSQL** 或 **MySQL** 等支持行级锁、多并发写入的数据库。 |
| **多服务器/分布式** | 多个物理服务器无法直接通过网络高效共享同一个本地 SQLite 文件。 | 使用网络型关系数据库（C/S 架构）或云数据库。 |
| **极其庞大的数据量** | 虽然理论上支持最大 281 TB 的单个文件，但当文件达到数百 GB 时，单文件维护、备份、索引重建、内存管理都会变得极其笨重。 | 换用分布式数据库，或者按时间/业务进行分库分表。 |
| **复杂的权限与安全性控制** | SQLite 文件没有任何用户权限和加密功能。任何人只要拿到了该文件，就可以读取和修改全部数据。 | 换用支持细粒度用户权限控制（GRANT/REVOKE）的网络数据库。 |

## 六、 总结

* **什么时候用 SQLite？**
  * 本地配置存储。
  * 桌面应用、移动端 App、物联网（IoT）嵌入式设备。
  * 数据分析、小型网站、测试环境。
  * 单机版向量存储底座（如本地的 **Chroma**，它使用 SQLite 保存向量之外的文档内容、ID 和 Metadata）。
* **什么时候不用？**
  * 多台服务器需要并发访问同一个数据库。
  * 有大量并发写操作的 Web 应用程序。
