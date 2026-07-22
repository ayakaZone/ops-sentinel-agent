"""向量索引服务模块"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.services.document_cleaner_service import document_cleaner_service
from app.services.document_loader_service import document_loader_service
from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


def compute_content_hash(text: str) -> str:
    """
    给一段文本算一个内容哈希值，用来判断"这段内容跟之前比有没有变"

    这里用 MD5 不是为了安全（不是存密码那种场景），只是用来快速判断两段文本
    是不是完全一样——只要输入哪怕改一个字符，算出来的值就会完全不同；
    输入完全相同，算出来的值必然完全相同。

    Args:
        text: 要计算哈希的文本（这里传入的是"清洗后"的文本，不是原始文本，
              原因见 app/services/vector_index_service.py 增量索引部分的注释）

    Returns:
        str: 32 位的十六进制哈希字符串，例如 "5d41402abc4b2a76b9719d911017c592"
    """
    # hashlib.md5() 要求传入 bytes（字节），不能直接传 Python 的 str（字符串），
    # 所以要先用 .encode("utf-8") 把字符串编码成字节；
    # .hexdigest() 把计算结果转成人类可读的十六进制字符串（不用 .digest()，
    # 那个返回的是不好直接打印/存储的原始字节）。
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        # 内容没变、被跳过的文件数——增量索引专门加的统计项，数值越高，
        # 说明这次同步省下的 embedding 调用越多。
        self.skipped_count = 0
        # 因为磁盘上文件已经被删除、而被清理掉的孤儿文件名列表。
        self.deleted_orphan_files: list[str] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def increment_skipped_count(self):
        """增加跳过计数（内容未变化）"""
        self.skipped_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件"""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取耗时（毫秒）"""
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "skipped_count": self.skipped_count,
            "deleted_orphan_files": self.deleted_orphan_files,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务 - 负责读取文件、生成向量、存储到 Milvus"""

    def __init__(self):
        """初始化向量索引服务"""
        self.upload_path = "./aiops-docs"
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """
        同步指定目录下的所有文件到 Milvus（增量索引 + 孤儿数据清理）

        跟"索引"不完全是一回事，这个方法做的其实是"同步"：
        1. 磁盘上有、库里没有 / 内容变了的文件 —— 索引或重新索引
        2. 磁盘上有、库里也有、内容没变的文件 —— 跳过，不产生 embedding 调用
        3. 库里有、磁盘上已经没有的文件 —— 视为已删除，清理掉对应的旧片段

        Args:
            directory_path: 目录路径（可选，默认使用知识库目录 aiops-docs）

        Returns:
            IndexingResult: 索引结果（含跳过数量、清理掉的孤儿文件列表）
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 使用指定目录或默认上传目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 获取所有支持的文件
            files = (
                list(dir_path.glob("*.txt"))
                + list(dir_path.glob("*.md"))
                + list(dir_path.glob("*.pdf"))
                + list(dir_path.glob("*.docx"))
            )

            # --- 第一步：孤儿数据清理 ---
            # 拿到"磁盘上现在实际存在的文件名"集合，跟"Milvus 里已经索引过的文件名"
            # 集合做差集运算：只在 Milvus 里出现、磁盘上已经没有的文件名，
            # 说明源文件被删掉了，对应的旧片段要清理掉，不然会变成永久检索得到、
            # 但实际源文件已经不存在的"僵尸数据"。
            disk_file_names = {f.name for f in files}
            indexed_file_names = vector_store_manager.list_indexed_file_names()
            # 集合的 "-" 运算符表示差集：indexed_file_names 里有、但 disk_file_names
            # 里没有的元素。None 也可能混进 indexed_file_names（老数据没有 _file_name
            # 字段时查出来是 None），这里顺手过滤掉。
            orphaned_file_names = {
                name for name in (indexed_file_names - disk_file_names) if name
            }
            for orphan_name in orphaned_file_names:
                deleted = vector_store_manager.delete_by_source(orphan_name)
                if deleted:
                    result.deleted_orphan_files.append(orphan_name)
                    logger.info(f"清理孤儿数据: {orphan_name}, 删除 {deleted} 个分片")

            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始同步目录: {target_path}, 找到 {len(files)} 个文件")

            # --- 第二步：逐个文件增量索引 ---
            for file_path in files:
                try:
                    # index_single_file 现在会返回 True/False，
                    # 表示这次调用是"真的重新索引了"还是"内容没变、跳过了"
                    was_indexed = self.index_single_file(str(file_path))
                    if was_indexed:
                        result.increment_success_count()
                        logger.info(f"✓ 文件索引成功: {file_path.name}")
                    else:
                        result.increment_skipped_count()
                        logger.info(f"- 文件内容未变化，跳过: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录同步完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 跳过={result.skipped_count}, "
                f"失败={result.fail_count}, 清理孤儿={len(result.deleted_orphan_files)}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str, force: bool = False) -> bool:
        """
        索引单个文件 (使用新的 LangChain 分割器)，内容没变时自动跳过

        Args:
            file_path: 文件路径
            force: 是否强制重新索引，忽略哈希比对结果（默认 False）。
                   平时不需要传这个参数；只有你明确知道"清洗逻辑改了、
                   想让所有文件都重新走一遍流程"时才用得上。

        Returns:
            bool: True 表示这次真的执行了索引（新文件或内容变了）；
                  False 表示内容没变，跳过了，没有产生任何 embedding 调用

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引失败时抛出
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始处理文件: {path}")

        try:
            # 1. 读取文件内容（按扩展名分发：.md/.txt 直接读，.pdf/.docx 各自用专门的加载器）
            content = document_loader_service.load(str(path))
            logger.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")

            # 2. 清洗文本（Unicode规范化/去不可见字符/去页眉页脚/段落去重）
            content = document_cleaner_service.clean(content)
            logger.info(f"文本清洗完成: {path}, 清洗后长度: {len(content)} 字符")

            # 3. 增量判断：算出这份清洗后内容的哈希值，跟 Milvus 里已存的比对。
            #    这一步特意放在"清洗完成之后"而不是"读取原始文件之后"——
            #    这样哪怕原始文件字节没变，只要以后改进了清洗逻辑（比如清洗规则更严格了），
            #    清洗完的文本会变、哈希值也会跟着自然变化，增量索引会正确地重新处理，
            #    不需要额外记得"这次要强制刷新"。
            content_hash = compute_content_hash(content)
            file_name = path.name

            if not force:
                existing_hash = vector_store_manager.get_content_hash(file_name)
                if existing_hash is not None and existing_hash == content_hash:
                    logger.info(f"文件内容未变化，跳过索引: {path}")
                    return False

            # 4. 删除该文件的旧数据（如果存在）——内容真的变了，或者是新文件
            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            # 5. 使用新的文档分割器（把这次算好的 content_hash 一起传进去，
            #    写进每个分片的 metadata，供下次增量同步比对用）
            documents = document_splitter_service.split_document(
                content, normalized_path, content_hash
            )
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 6. 添加文档到向量存储
            if documents:
                vector_store_manager.add_documents(documents)
                logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片")
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

            return True

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e


# 全局单例
vector_index_service = VectorIndexService()
