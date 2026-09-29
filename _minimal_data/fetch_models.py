"""
从 HF 镜像 (aifasthub.com) 把模型文件拉到本地目录，供 from_pretrained 直接加载。

为什么要自己写而不用 huggingface_hub：
  1. 镜像的部分文件 size 元数据错误，会触发 hub 的一致性校验失败；
  2. chatglm/uie 那类代码硬编码了 huggingface.co 地址，根本不走 hub。

关键点：镜像会 302 重定向到 us.aws.cdn.hf.co，那条连接容易被提前掐断，
       所以必须用 curl 下载 + 下载后校验大小 + 失败重试。

产物: D:/new plm/_models/<repo_name>/
"""
import json
import os
import subprocess
import sys
import urllib.request

MIRROR = 'https://aifasthub.com'
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_models')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Accept': '*/*',
}

# 只下 PyTorch 权重 + 配置。跳过 README/图片，以及 TensorFlow / Flax 的冗余权重
# （tf_model.h5 / flax_model.msgpack 每个 400~850MB，PyTorch 用不到，纯浪费带宽）
SKIP_SUFFIX = ('.md', '.gitattributes', '.png', '.jpg', '.jpeg', '.gif',
               '.h5', '.msgpack', '.ot')

REPOS = [
    'nghuyong/ernie-3.0-base-zh',
    'uer/t5-base-chinese-cluecorpussmall',
    'uer/gpt2-chinese-cluecorpussmall',
    'uer/roberta-base-finetuned-jd-binary-chinese',
]


def list_files(repo):
    """返回 [(filename, size), ...]"""
    url = f'{MIRROR}/api/models/{repo}?blobs=true'
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    out = []
    for s in data.get('siblings', []):
        name = s.get('rfilename', '')
        if not name or name.endswith(SKIP_SUFFIX):
            continue
        out.append((name, s.get('size')))
    return out


def curl_download(url, dest):
    """用 curl 下载，支持断点续传和重试。"""
    cmd = [
        'curl', '-L', '--fail', '--silent', '--show-error',
        '--retry', '6', '--retry-delay', '5', '--retry-all-errors',
        '--connect-timeout', '30', '--max-time', '3600',
        '-C', '-',                       # 断点续传
        '-A', HEADERS['User-Agent'],
        '-o', dest, url,
    ]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def fetch(repo, filename, expected_size, dest_dir):
    url = f'{MIRROR}/{repo}/resolve/main/{filename}'
    dest = os.path.join(dest_dir, filename)

    if expected_size and os.path.exists(dest) and os.path.getsize(dest) == expected_size:
        print(f'    [skip] {filename} ({expected_size/1e6:.1f} MB, 已完整)', flush=True)
        return True

    for attempt in range(1, 5):
        ok = curl_download(url, dest)
        actual = os.path.getsize(dest) if os.path.exists(dest) else 0
        if ok and expected_size and actual == expected_size:
            print(f'    [ok]   {filename}  {actual/1e6:.1f} MB', flush=True)
            return True
        if ok and not expected_size:
            print(f'    [ok]   {filename}  {actual/1e6:.1f} MB (无预期大小)', flush=True)
            return True
        print(f'    [retry {attempt}/4] {filename}  '
              f'期望 {expected_size} 实际 {actual}', flush=True)

    print(f'    [FAIL] {filename} 重试 4 次仍不完整', flush=True)
    return False


def main():
    for repo in REPOS:
        name = repo.split('/')[-1]
        dest_dir = os.path.join(ROOT, name)
        os.makedirs(dest_dir, exist_ok=True)
        print(f'==> {repo}', flush=True)
        try:
            files = list_files(repo)
        except Exception as e:
            print(f'    [FAIL] 无法获取文件列表: {e}', flush=True)
            continue
        ok = sum(1 for fn, sz in files if fetch(repo, fn, sz, dest_dir))
        print(f'    完成 {ok}/{len(files)}  ->  {dest_dir}', flush=True)
    print('[ALL DONE]', flush=True)


if __name__ == '__main__':
    main()
