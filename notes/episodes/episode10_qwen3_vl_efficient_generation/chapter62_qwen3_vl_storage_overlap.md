# Chapter 62 — Qwen3-VL cache 重叠判断不再调用 dataids

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

稳态 decode 的 GPU 分配已经是 0，但一步仍有约 `7.59 MB` 的 Julia 对象。
其中每次校验都会对 56 块 K/V 做三角比较，`Base.mightalias` 里的 `dataids`
和 `pointer` 各是一个 16 byte 对象。

重叠判断改成：同一父对象才可能重叠；dense 数组用对象身份；`SubArray`
只比较 unit-range 下标。Chapter 46 的跨层重叠与不相交 view 仍按原错误
失败或通过。

这一改动和 Chapter 63—67 的元素 kernel 一起测。联合关闭采样见 Chapter 67。
