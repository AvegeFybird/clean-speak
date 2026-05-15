from dataclasses import dataclass


# 存放数据的类
@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    expansion_terms: list[str]


@dataclass(frozen=True)
class RewriteRule:
    triggers: tuple[str,...]
    expansion_terms: tuple[str,...]


REWRITE_RULES: tuple[RewriteRule,...] = (
     RewriteRule(
        triggers=("不出水", "不喷水", "拖布不湿", "出水少"),
        expansion_terms=("水箱", "出水管", "出水量", "拖布"),
    ),
    RewriteRule(
        triggers=("吸力变弱", "吸力突然变弱", "吸不干净", "吸力小"),
        expansion_terms=("尘盒", "滤网", "风道", "主刷"),
    ),
    RewriteRule(
        triggers=("宠物", "猫毛", "狗毛", "毛发", "头发"),
        expansion_terms=("毛发", "防缠绕", "滤网", "主刷"),
    ),
    RewriteRule(
        triggers=("维护", "保养", "清理", "清洁"),
        expansion_terms=("尘盒", "滤网", "边刷", "主刷"),
    ),
    RewriteRule(
        triggers=("选购", "怎么选", "推荐", "小户型"),
        expansion_terms=("续航", "吸力", "导航", "避障"),
    ),
    RewriteRule(
        triggers=("扫拖", "拖地", "扫拖一体"),
        expansion_terms=("水箱", "拖布", "出水量", "地面"),
    ),
    RewriteRule(
        triggers=("老人", "长辈", "父母"),
        expansion_terms=("按键简洁", "语音控制", "APP界面", "操作简单"),
    ),
    RewriteRule(
        triggers=("地毯", "禁拖"),
        expansion_terms=("地毯识别", "地毯增压", "湿拖禁区", "拖布抬升"),
    ),
    RewriteRule(
        triggers=("边角", "沿边"),
        expansion_terms=("边刷", "沿边清扫", "沿边传感器", "边角增强"),
    ),
)


def rewrite_query(query: str) -> RewriteResult:
    original_query = query = query.strip()
    expansion_terms: list[str] = []

    for rule in REWRITE_RULES:
        if any(trigger in query for trigger in rule.triggers):
            for term in rule.expansion_terms:
                if term not in expansion_terms and term not in original_query:
                    expansion_terms.append(term)

    rewritten_query = original_query
    if expansion_terms:
        rewritten_query = f"{original_query} {' '.join(expansion_terms)}"

    return RewriteResult(
        original_query=original_query,
        rewritten_query=rewritten_query,
        expansion_terms=expansion_terms,
    )
                
