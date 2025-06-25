import pandas as pd
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import tempfile
import os
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_video_script_enhanced(theme, length, creativity, api_key, base_url="https://api.openai-hk.com/v1",
                                   style="科普教育", audience="", hooks=True, cta=True, model="gpt-4o-mini",
                                   temperature=0.7):
    """
    增强版视频脚本生成器
    """
    try:
        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model, base_url=base_url)

        # 构建详细的提示词
        style_prompts = {
            "科普教育": "采用清晰易懂的语言，循序渐进地解释概念，适合教育类视频",
            "娱乐搞笑": "使用幽默风趣的语言，增加互动元素和笑点",
            "商业营销": "突出产品价值，包含明确的行动号召和转化点",
            "纪录片": "采用客观叙述风格，注重事实和细节描述",
            "新闻播报": "使用正式新闻语言，结构清晰，重点突出",
            "个人分享": "采用亲切自然的语调，增加个人经历和感受"
        }

        style_desc = style_prompts.get(style, style_prompts["科普教育"])

        template = f"""你是一位专业的视频脚本撰写者，擅长{style}风格。

请为主题：{theme}，时长：{length}分钟的视频创作一个完整的脚本。

要求：
1. 风格：{style_desc}
2. 目标受众：{audience if audience else "一般观众"}
3. 时长控制：{length}分钟（约{int(length * 150)}字）
4. 创造力：{creativity}（0=严谨，1=创意）
5. 开场钩子：{'需要' if hooks else '不需要'}
6. 行动号召：{'需要' if cta else '不需要'}

请按以下格式输出：
# 视频标题
[标题]

# 开场钩子
[吸引观众的开场]

# 主要内容
[分段落的主要内容，每段标注时长]

# 结尾总结
[总结和行动号召]

# 拍摄建议
[镜头、场景、道具等建议]

请确保内容结构清晰，语言生动有趣。"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(theme=theme, length=length)

        return result

    except Exception as e:
        raise Exception(f"视频脚本生成失败：{str(e)}")


def generate_xiaohongshu_content_enhanced(theme, api_key, base_url="https://api.openai-hk.com/v1",
                                          content_type="种草推荐", tone="亲切自然", num_variations=5,
                                          audience="", hashtags=True, emoji=True, model="gpt-4o-mini",
                                          temperature=0.7):
    """
    增强版小红书文案生成器
    """
    try:
        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model, base_url=base_url)

        # 内容类型提示
        type_prompts = {
            "种草推荐": "突出产品优势，包含使用体验和推荐理由",
            "经验分享": "分享个人经历和心得，增加实用价值",
            "生活记录": "记录日常生活片段，增加情感共鸣",
            "知识科普": "传播专业知识，语言通俗易懂",
            "情感故事": "讲述情感经历，增加情感共鸣",
            "美食探店": "描述美食体验，包含环境、味道、价格等"
        }

        # 语调提示
        tone_prompts = {
            "亲切自然": "像朋友聊天一样自然亲切",
            "专业权威": "专业可信，有权威性",
            "活泼可爱": "年轻活力，充满正能量",
            "文艺清新": "文艺范儿，清新脱俗",
            "幽默风趣": "幽默有趣，增加笑点"
        }

        type_desc = type_prompts.get(content_type, type_prompts["种草推荐"])
        tone_desc = tone_prompts.get(tone, tone_prompts["亲切自然"])

        template = f"""你是一位小红书爆款写手，擅长{content_type}类型的内容创作。

请为主题：{theme} 创作{num_variations}个不同版本的文案。

要求：
1. 内容类型：{type_desc}
2. 语调风格：{tone_desc}
3. 目标用户：{audience if audience else "小红书用户"}
4. 包含表情符号：{'是' if emoji else '否'}
5. 包含话题标签：{'是' if hashtags else '否'}
6. 每个文案包含：标题 + 正文 + 标签

请按以下格式输出：

## 文案1
**标题：** [吸引人的标题]
**正文：** [正文内容]
**标签：** [相关话题标签]

## 文案2
**标题：** [吸引人的标题]
**正文：** [正文内容]
**标签：** [相关话题标签]

[继续其他文案...]

请确保每个文案都有不同的角度和表达方式，避免重复。"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(theme=theme)

        return result

    except Exception as e:
        raise Exception(f"小红书文案生成失败：{str(e)}")


def chat_with_pdf_enhanced(file, question, api_key, base_url="https://api.openai-hk.com/v1",
                           model="gpt-4o-mini", temperature=0.0):
    """
    增强版PDF问答系统
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # 加载PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        # 文本分割
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(pages)

        # 提取文本内容
        context_parts = []
        for text in texts:
            content = text.page_content.strip()
            if content and len(content) > 50:  # 过滤太短的内容
                context_parts.append(content)

        context = "\n\n".join(context_parts)

        # 限制 context 长度
        max_context_chars = 4000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n\n【内容过长，仅截取部分参与问答】"

        # 创建LLM
        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model, base_url=base_url)

        # 构建智能提示词
        template = """你是一位专业的PDF智能问答助手。请根据以下PDF内容回答用户问题。

PDF内容：
{context}

用户问题：{question}

回答要求：
1. 基于PDF内容准确回答
2. 如果PDF中没有相关信息，请明确说明
3. 回答要简洁明了，结构清晰
4. 可以适当补充相关知识，但要标注来源
5. 使用中文回答

请用专业、友好的语调回答："""

        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(context=context, question=question)

        # 清理临时文件
        os.unlink(tmp_path)

        return answer

    except Exception as e:
        # 确保清理临时文件
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise Exception(f"PDF问答失败：{str(e)}")


def analyze_csv_with_plot_enhanced(df, query, api_key, base_url="https://api.openai-hk.com/v1",
                                   model="gpt-4o-mini", temperature=0.0):
    """
    增强版CSV分析工具
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return "", "CSV文件为空或格式错误，请检查后重新上传。", None

        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model, base_url=base_url)

        # 数据预处理信息
        df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'sample_data': df.head(5).to_dict('records')
        }

        template = """
你是一位数据分析和可视化专家。用户会给你一个数据分析或可视化请求，请你返回三部分内容：

1. 画图代码（只返回可直接用exec执行的python代码，不要返回markdown代码块）
2. 分析文本（简要说明图表含义和数据洞察）
3. 图表类型（返回图表类型名称，如：折线图、柱状图、散点图等）

数据信息：
- 数据形状：{shape}
- 列名：{columns}
- 数值列：{numeric_columns}
- 分类列：{categorical_columns}
- 缺失值：{null_counts}
- 数据预览：{sample_data}

用户请求：{query}

请严格按如下格式返回：
[CODE]
# 画图代码
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
...
[ENDCODE]
[ANALYSIS]
# 分析文本
...
[ENDANALYSIS]
[CHART_TYPE]
# 图表类型
...
[ENDCHART_TYPE]
"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            shape=df_info['shape'],
            columns=df_info['columns'],
            numeric_columns=df_info['numeric_columns'],
            categorical_columns=df_info['categorical_columns'],
            null_counts=df_info['null_counts'],
            sample_data=df_info['sample_data'],
            query=query
        )

        # 解析返回内容
        code = ""
        analysis = ""
        chart_type = ""

        if "[CODE]" in response and "[ENDCODE]" in response:
            code = response.split("[CODE]")[1].split("[ENDCODE]")[0].strip()

        if "[ANALYSIS]" in response and "[ENDANALYSIS]" in response:
            analysis = response.split("[ANALYSIS]")[1].split("[ENDANALYSIS]")[0].strip()

        if "[CHART_TYPE]" in response and "[ENDCHART_TYPE]" in response:
            chart_type = response.split("[CHART_TYPE]")[1].split("[ENDCHART_TYPE]")[0].strip()

        return code, analysis, chart_type

    except Exception as e:
        raise Exception(f"CSV分析失败：{str(e)}")


def chat_with_ai_enhanced(input_text, api_key, base_url="https://api.openai-hk.com/v1",
                          chat_history=None, mode="通用助手", model="gpt-4o-mini", temperature=0.7):
    """
    增强版AI对话系统
    """
    try:
        llm = ChatOpenAI(temperature=temperature, openai_api_key=api_key, model_name=model, base_url=base_url)

        # 模式提示词
        mode_prompts = {
            "通用助手": "你是一位全能AI助手，能够回答各种问题，提供帮助和建议。",
            "编程专家": "你是一位编程专家，擅长各种编程语言和技术问题，能够提供代码示例和技术指导。",
            "写作助手": "你是一位写作助手，擅长文案创作、文章写作、内容优化等。",
            "学习导师": "你是一位学习导师，能够提供学习方法、知识讲解、学习规划等指导。",
            "创意伙伴": "你是一位创意伙伴，能够提供创意灵感、头脑风暴、创新思维等帮助。"
        }

        mode_desc = mode_prompts.get(mode, mode_prompts["通用助手"])

        # 构建对话历史
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_parts = []
            for msg in chat_history[-10:]:  # 只保留最近10轮对话
                role = "用户" if msg['role'] == '用户' else "AI助手"
                history_parts.append(f"{role}: {msg['content']}")
            history_text = "\n".join(history_parts) + "\n\n"

        template = f"""{mode_desc}

{history_text}用户: {input_text}
AI助手: """

        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(input_text=input_text)

        return response

    except Exception as e:
        raise Exception(f"AI对话失败：{str(e)}")


# 保留原有函数以保持兼容性
def generate_video_script(theme, length, creativity, api_key, base_url="https://api.openai-hk.com/v1"):
    return generate_video_script_enhanced(theme, length, creativity, api_key, base_url)


def generate_xiaohongshu_content(theme, api_key, base_url="https://api.openai-hk.com/v1"):
    return generate_xiaohongshu_content_enhanced(theme, api_key, base_url)


def chat_with_pdf(file, question, api_key, base_url="https://api.openai-hk.com/v1"):
    return chat_with_pdf_enhanced(file, question, api_key, base_url)


def analyze_csv(df, query, api_key, base_url="https://api.openai-hk.com/v1"):
    code, analysis, _ = analyze_csv_with_plot_enhanced(df, query, api_key, base_url)
    return analysis, df


def chat_with_ai(input_text, api_key, base_url="https://api.openai-hk.com/v1"):
    return chat_with_ai_enhanced(input_text, api_key, base_url)


def analyze_csv_with_plot(df, query, api_key, base_url="https://api.openai-hk.com/v1"):
    code, analysis, _ = analyze_csv_with_plot_enhanced(df, query, api_key, base_url)
    return code, analysis