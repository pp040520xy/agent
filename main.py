import streamlit as st
import utils
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# 页面配置
st.set_page_config(
    page_title="自助全能AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<div class="main-header">
    <h1>🤖 自助全能AI</h1>
    <p style="font-size: 1.2rem; margin-top: 0;">集成视频脚本生成、小红书文案、PDF问答、CSV分析、AI对话五大功能，一站式AI体验</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    st.markdown("### 🔑 API 配置")
    openai_api_key = st.text_input('OpenAI API 密钥', type='password', key='api_key_input')

    # API状态检查
    if openai_api_key:
        if len(openai_api_key) > 20:
            st.success("✅ API密钥格式正确")
        else:
            st.error("❌ API密钥格式可能不正确")

    st.markdown('[获取OpenAI密钥](https://openai-hk.com/v3/ai/)')

    st.markdown("---")
    st.markdown("### 🎯 功能导航")

    # 功能统计
    if 'usage_stats' not in st.session_state:
        st.session_state.usage_stats = {
            'video_scripts': 0,
            'xhs_content': 0,
            'pdf_qa': 0,
            'csv_analysis': 0,
            'ai_chat': 0
        }

    st.markdown(f"""
    - 🎬 视频脚本生成 ({st.session_state.usage_stats['video_scripts']})
    - 📕 小红书文案 ({st.session_state.usage_stats['xhs_content']})
    - 📑 PDF问答 ({st.session_state.usage_stats['pdf_qa']})
    - 💡 CSV分析 ({st.session_state.usage_stats['csv_analysis']})
    - 🤖 AI对话 ({st.session_state.usage_stats['ai_chat']})
    """)

    st.markdown("---")
    st.markdown("### ⚙️ 设置")

    # 模型选择
    model_choice = st.selectbox(
        "选择AI模型",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )

    # 温度设置
    temperature = st.slider("AI创造力", 0.0, 1.0, 0.7, 0.1)

    # 清除历史记录
    if st.button("🗑️ 清除所有历史记录"):
        st.session_state.chat_history = []
        st.session_state.usage_stats = {
            'video_scripts': 0,
            'xhs_content': 0,
            'pdf_qa': 0,
            'csv_analysis': 0,
            'ai_chat': 0
        }
        st.success("历史记录已清除！")
        st.rerun()

base_url = "https://api.openai-hk.com/v1"

# 创建标签页
tabs = st.tabs([
    "🎬 视频脚本生成",
    "📕 小红书文案",
    "📑 PDF智能问答",
    "💡 CSV智能分析",
    "🤖 AI对话"
])


# 1. 视频脚本生成
def video_tab():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader('🎬 视频脚本生成器')
    st.markdown("**专业级视频脚本创作，支持多种风格和时长**")

    col1, col2 = st.columns(2)

    with col1:
        subject = st.text_input('📝 视频主题', placeholder="例如：人工智能的未来发展", key='video_subject')
        video_length = st.number_input('⏱️ 视频时长(分钟)', min_value=0.5, max_value=60.0, value=3.0, step=0.5,
                                       key='video_length')

    with col2:
        video_style = st.selectbox(
            '🎨 视频风格',
            ['科普教育', '娱乐搞笑', '商业营销', '纪录片', '新闻播报', '个人分享'],
            key='video_style'
        )
        creativity = st.slider('🎭 创造力', 0.0, 1.0, 0.7, 0.1, key='video_creativity',
                               help="0=严谨专业，1=创意多样")

    # 高级选项
    with st.expander("🔧 高级选项"):
        target_audience = st.text_input('👥 目标受众', placeholder="例如：18-35岁科技爱好者", key='target_audience')
        include_hooks = st.checkbox('🎣 包含开场钩子', value=True, key='include_hooks')
        call_to_action = st.checkbox('📢 包含行动号召', value=True, key='call_to_action')

    if st.button('🚀 生成脚本', key='video_btn', use_container_width=True):
        if not openai_api_key:
            st.markdown('<div class="error-box">请输入API密钥</div>', unsafe_allow_html=True)
            return
        if not subject:
            st.markdown('<div class="error-box">请输入视频主题</div>', unsafe_allow_html=True)
            return

        with st.spinner('🤖 AI正在创作专业脚本...'):
            try:
                result = utils.generate_video_script_enhanced(
                    subject, video_length, creativity, openai_api_key,
                    base_url=base_url, style=video_style,
                    audience=target_audience, hooks=include_hooks,
                    cta=call_to_action, model=model_choice, temperature=temperature
                )
                st.session_state.usage_stats['video_scripts'] += 1

                st.markdown('<div class="success-box">🎉 脚本生成成功！</div>', unsafe_allow_html=True)

                # 显示结果
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**📄 脚本内容：**")
                    st.markdown(result)

                with col2:
                    st.markdown("**📊 生成统计：**")
                    st.metric("脚本字数", len(result))
                    st.metric("预计时长", f"{video_length}分钟")
                    st.metric("使用次数", st.session_state.usage_stats['video_scripts'])

                    # 下载按钮
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 下载脚本",
                        data=result,
                        file_name=f"视频脚本_{subject}_{timestamp}.md",
                        mime="text/markdown"
                    )

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ 生成失败：{str(e)}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# 2. 小红书文案
def xhs_tab():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader('📕 小红书爆款文案生成器')
    st.markdown("**一键生成多个爆款标题和正文，提升内容传播力**")

    col1, col2 = st.columns(2)

    with col1:
        theme = st.text_input('📝 文案主题', placeholder="例如：护肤心得分享", key='xhs_theme')
        content_type = st.selectbox(
            '📄 内容类型',
            ['种草推荐', '经验分享', '生活记录', '知识科普', '情感故事', '美食探店'],
            key='content_type'
        )

    with col2:
        tone = st.selectbox(
            '🎭 文案语调',
            ['亲切自然', '专业权威', '活泼可爱', '文艺清新', '幽默风趣'],
            key='xhs_tone'
        )
        num_variations = st.slider('📊 生成数量', 1, 10, 5, key='num_variations')

    # 高级选项
    with st.expander("🔧 高级选项"):
        target_audience = st.text_input('👥 目标用户', placeholder="例如：25-35岁女性", key='xhs_audience')
        include_hashtags = st.checkbox('🏷️ 包含话题标签', value=True, key='include_hashtags')
        include_emoji = st.checkbox('😊 包含表情符号', value=True, key='include_emoji')

    if st.button('🚀 生成文案', key='xhs_btn', use_container_width=True):
        if not openai_api_key:
            st.markdown('<div class="error-box">请输入API密钥</div>', unsafe_allow_html=True)
            return
        if not theme:
            st.markdown('<div class="error-box">请输入文案主题</div>', unsafe_allow_html=True)
            return

        with st.spinner('🤖 AI正在创作爆款文案...'):
            try:
                result = utils.generate_xiaohongshu_content_enhanced(
                    theme, openai_api_key, base_url=base_url,
                    content_type=content_type, tone=tone,
                    num_variations=num_variations, audience=target_audience,
                    hashtags=include_hashtags, emoji=include_emoji,
                    model=model_choice, temperature=temperature
                )
                st.session_state.usage_stats['xhs_content'] += 1

                st.markdown('<div class="success-box">🎉 文案生成成功！</div>', unsafe_allow_html=True)

                # 显示结果
                st.markdown("**📄 文案内容：**")
                st.markdown(result)

                # 统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("生成数量", num_variations)
                with col2:
                    st.metric("文案字数", len(result))
                with col3:
                    st.metric("使用次数", st.session_state.usage_stats['xhs_content'])

                # 下载按钮
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 下载文案",
                    data=result,
                    file_name=f"小红书文案_{theme}_{timestamp}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ 生成失败：{str(e)}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# 3. PDF智能问答
def pdf_tab():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader('📑 PDF智能问答系统')
    st.markdown("**上传PDF文档，AI智能理解并回答相关问题**")

    uploaded_file = st.file_uploader(
        '📁 上传PDF文件',
        type=['pdf'],
        key='pdf_file',
        help="支持中文PDF文档，最大文件大小10MB"
    )

    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📄 文件大小: {file_size:.2f} MB")

        # 显示文件信息
        col1, col2 = st.columns(2)
        with col1:
            st.metric("文件名", uploaded_file.name)
        with col2:
            st.metric("文件大小", f"{file_size:.2f} MB")

    # 预设问题
    preset_questions = [
        "请总结这篇文档的主要内容",
        "文档中提到了哪些关键概念？",
        "请列出文档中的主要观点",
        "文档的结论是什么？",
        "请解释文档中的专业术语"
    ]

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input('❓ 请输入你的问题', key='pdf_question')
    with col2:
        if st.button('🎲 随机问题'):
            import random
            question = random.choice(preset_questions)
            st.session_state.pdf_question = question
            st.rerun()

    # 显示预设问题
    st.markdown("**💡 预设问题：**")
    for i, q in enumerate(preset_questions, 1):
        if st.button(f"{i}. {q}", key=f"preset_q_{i}"):
            st.session_state.pdf_question = q
            st.rerun()

    if st.button('🔍 开始问答', key='pdf_btn', use_container_width=True):
        if not openai_api_key:
            st.markdown('<div class="error-box">请输入API密钥</div>', unsafe_allow_html=True)
            return
        if not uploaded_file:
            st.markdown('<div class="error-box">请上传PDF文件</div>', unsafe_allow_html=True)
            return
        if not question:
            st.markdown('<div class="error-box">请输入问题</div>', unsafe_allow_html=True)
            return

        with st.spinner('🤖 AI正在分析PDF并回答问题...'):
            try:
                answer = utils.chat_with_pdf_enhanced(
                    uploaded_file, question, openai_api_key,
                    base_url=base_url, model=model_choice, temperature=temperature
                )
                st.session_state.usage_stats['pdf_qa'] += 1

                st.markdown('<div class="success-box">🎉 回答完成！</div>', unsafe_allow_html=True)

                # 显示答案
                st.markdown("**💬 AI回答：**")
                st.markdown(answer)

                # 统计信息
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("回答字数", len(answer))
                with col2:
                    st.metric("使用次数", st.session_state.usage_stats['pdf_qa'])

            except Exception as e:
                st.markdown(f'<div class="error-box">❌ 问答失败：{str(e)}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# 4. CSV智能分析
def csv_tab():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader('💡 CSV数据分析智能工具')
    st.markdown("**上传CSV文件，AI自动分析数据并生成可视化图表**")

    uploaded_csv = st.file_uploader(
        '📁 上传CSV文件',
        type=['csv'],
        key='csv_file',
        help="支持标准CSV格式，建议文件大小不超过50MB"
    )

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)

            # 数据概览
            st.markdown("**📊 数据概览：**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("行数", len(df))
            with col2:
                st.metric("列数", len(df.columns))
            with col3:
                st.metric("缺失值", df.isnull().sum().sum())
            with col4:
                st.metric("数据类型", len(df.dtypes.unique()))

            # 数据预览
            with st.expander("👀 数据预览"):
                st.dataframe(df.head(10), use_container_width=True)

            # 数据类型信息
            with st.expander("📋 数据类型信息"):
                dtype_info = pd.DataFrame({
                    '列名': df.columns,
                    '数据类型': df.dtypes,
                    '非空值数量': df.count(),
                    '缺失值数量': df.isnull().sum()
                })
                st.dataframe(dtype_info, use_container_width=True)

            # 分析请求
            st.markdown("**🔍 分析请求：**")

            # 预设分析
            preset_analyses = [
                "请分析数据的整体分布情况",
                "生成数据相关性分析图表",
                "创建数据趋势分析图",
                "分析异常值和离群点",
                "生成数据统计摘要"
            ]

            selected_analysis = st.selectbox("选择预设分析", ["自定义分析"] + preset_analyses)

            if selected_analysis == "自定义分析":
                query = st.text_area(
                    '📝 请输入你的分析需求/问题/可视化请求',
                    placeholder="例如：请分析销售数据的月度趋势，并生成折线图",
                    key='csv_query'
                )
            else:
                query = selected_analysis
                st.text_area("当前分析请求", query, key='csv_query')

            if st.button('🔍 开始分析', key='csv_btn', use_container_width=True):
                if not openai_api_key:
                    st.markdown('<div class="error-box">请输入API密钥</div>', unsafe_allow_html=True)
                    return
                if not query:
                    st.markdown('<div class="error-box">请输入分析需求</div>', unsafe_allow_html=True)
                    return

                with st.spinner('🤖 AI正在分析数据...'):
                    try:
                        code, analysis, chart_type = utils.analyze_csv_with_plot_enhanced(
                            df, query, openai_api_key, base_url=base_url,
                            model=model_choice, temperature=temperature
                        )
                        st.session_state.usage_stats['csv_analysis'] += 1

                        st.markdown('<div class="success-box">🎉 分析完成！</div>', unsafe_allow_html=True)

                        # 显示分析结果
                        st.markdown("**📈 分析结果：**")
                        st.markdown(analysis)

                        # 显示图表
                        if code and chart_type:
                            st.markdown("**📊 可视化图表：**")
                            try:
                                # 创建安全的执行环境
                                safe_dict = {
                                    'df': df, 'px': px, 'go': go, 'st': st,
                                    'pd': pd, 'plt': plt
                                }
                                exec(code, {"__builtins__": {}}, safe_dict)
                            except Exception as e:
                                st.error(f'图表生成失败：{e}')
                                st.code(code, language='python')

                        # 统计信息
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("数据行数", len(df))
                        with col2:
                            st.metric("分析字数", len(analysis))
                        with col3:
                            st.metric("使用次数", st.session_state.usage_stats['csv_analysis'])

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ 分析失败：{str(e)}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"CSV文件读取失败：{e}")
    else:
        st.info('📁 请上传CSV文件以开始分析')

    st.markdown('</div>', unsafe_allow_html=True)


# 5. AI对话
def chat_tab():
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader('🤖 通用AI对话助手')
    st.markdown("**智能对话，支持多轮对话和上下文记忆**")

    # 初始化聊天历史
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # 对话设置
    col1, col2 = st.columns(2)
    with col1:
        chat_mode = st.selectbox(
            "🎭 对话模式",
            ["通用助手", "编程专家", "写作助手", "学习导师", "创意伙伴"],
            key='chat_mode'
        )
    with col2:
        max_history = st.slider("📚 记忆轮数", 1, 20, 10, key='max_history')

    # 聊天输入
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_area(
            '💬 请输入你的问题',
            placeholder="你好！我是你的AI助手，有什么可以帮助你的吗？",
            key='chat_input',
            height=100
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submitted = st.form_submit_button('🚀 发送', use_container_width=True)
        with col2:
            if st.form_submit_button('🗑️ 清空历史', use_container_width=True):
                st.session_state['chat_history'] = []
                st.rerun()
        with col3:
            if st.form_submit_button('📥 导出对话', use_container_width=True):
                if st.session_state['chat_history']:
                    chat_text = "\n\n".join([
                        f"{msg['role']}: {msg['content']}"
                        for msg in st.session_state['chat_history']
                    ])
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="下载对话记录",
                        data=chat_text,
                        file_name=f"AI对话记录_{timestamp}.txt",
                        mime="text/plain"
                    )

        if submitted:
            if not openai_api_key:
                st.markdown('<div class="error-box">请输入API密钥</div>', unsafe_allow_html=True)
            elif not user_input:
                st.markdown('<div class="error-box">请输入问题</div>', unsafe_allow_html=True)
            else:
                with st.spinner('🤖 AI正在思考...'):
                    try:
                        response = utils.chat_with_ai_enhanced(
                            user_input, openai_api_key, base_url=base_url,
                            chat_history=st.session_state['chat_history'][-max_history * 2:],
                            mode=chat_mode, model=model_choice, temperature=temperature
                        )
                        st.session_state.usage_stats['ai_chat'] += 1

                        # 添加对话记录
                        st.session_state['chat_history'].append({'role': '用户', 'content': user_input})
                        st.session_state['chat_history'].append({'role': 'AI助手', 'content': response})

                        # 限制历史记录长度
                        if len(st.session_state['chat_history']) > max_history * 2:
                            st.session_state['chat_history'] = st.session_state['chat_history'][-max_history * 2:]

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ 对话失败：{str(e)}</div>', unsafe_allow_html=True)

    # 显示聊天历史
    st.markdown("---")
    st.markdown("**💬 聊天历史：**")

    if st.session_state['chat_history']:
        for i, msg in enumerate(st.session_state['chat_history']):
            if msg['role'] == '用户':
                with st.chat_message("user"):
                    st.markdown(msg['content'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg['content'])
    else:
        st.info("💬 开始你的第一次对话吧！")

    # 统计信息
    if st.session_state['chat_history']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("对话轮数", len(st.session_state['chat_history']) // 2)
        with col2:
            total_chars = sum(len(msg['content']) for msg in st.session_state['chat_history'])
            st.metric("总字符数", total_chars)
        with col3:
            st.metric("使用次数", st.session_state.usage_stats['ai_chat'])

    st.markdown('</div>', unsafe_allow_html=True)


# 标签页分发
with tabs[0]:
    video_tab()
with tabs[1]:
    xhs_tab()
with tabs[2]:
    pdf_tab()
with tabs[3]:
    csv_tab()
with tabs[4]:
    chat_tab()

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 全能AI助手 Pro | 让AI为你服务 | 版本 2.0</p>
    <p>如有问题请刷新页面或检查API配置</p>
</div>
""", unsafe_allow_html=True)