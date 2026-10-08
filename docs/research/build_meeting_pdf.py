"""Build the concise Korean research meeting PDF with embedded fonts.

Uses ReportLab; defaults to Malgun Gothic on Windows. Override PDF_FONT_REGULAR
and PDF_FONT_BOLD to other Korean TrueType fonts when building elsewhere.
"""
from pathlib import Path
import os
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/BadFU_research_meeting_2026-10-08.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("Korean", os.environ.get("PDF_FONT_REGULAR", "C:/Windows/Fonts/malgun.ttf")))
pdfmetrics.registerFont(TTFont("KoreanBold", os.environ.get("PDF_FONT_BOLD", "C:/Windows/Fonts/malgunbd.ttf")))
pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="KoreanBold")

W, H = A4
M, CW = 42, W - 84
NAVY = colors.HexColor("#142F40")
TEAL = colors.HexColor("#047F7C")
INK = colors.HexColor("#273B48")
MUTED = colors.HexColor("#61737E")
PALE = colors.HexColor("#EBF5F3")
GRAY = colors.HexColor("#F2F5F7")
LINE = colors.HexColor("#DCE5E9")
WHITE = colors.white
c = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
c.setTitle("삭제 요청 이행과 백도어 정화 - 연구 미팅 핵심 요약")
c.setAuthor("BadFU Defend Research")
c.setSubject("알고리즘, 예비 근거, GPU 검증 계획과 선행연구")
c.setCreator("ReportLab / BadFU research meeting brief")
c.setViewerPreference("DisplayDocTitle", "true")

def box(x, y, w, h, fill=GRAY, radius=8, stroke=None):
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, H-y-h, w, h, radius, stroke=bool(stroke), fill=1)

def line(x1, y1, x2, y2, color=LINE, width=0.7):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x1, H-y1, x2, H-y2)

def text(s, x, y, w=CW, size=10, leading=None, color=INK, bold=False,
         maxh=200, align=0):
    st = ParagraphStyle("p", fontName="KoreanBold" if bold else "Korean",
                        fontSize=size, leading=leading or size*1.55,
                        textColor=color, wordWrap=None, alignment=align,
                        spaceBefore=0, spaceAfter=0, borderPadding=0)
    p = Paragraph(s, st)
    pw, ph = p.wrap(w, H)
    if ph > maxh + .05:
        raise ValueError(f"Text overflow {ph:.2f}>{maxh}: {s[:80]}")
    if y + ph > H-40:
        raise ValueError(f"Footer collision: {s[:80]}")
    p.drawOn(c, x, H-y-ph)
    return ph

def heading(s, y, n=None):
    if n:
        text(n, M, y+1, 26, 9, 13, TEAL, True, maxh=18)
        text(s, M+30, y, CW-30, 14, 20, NAVY, True, maxh=24)
    else:
        text(s, M, y, CW, 14, 20, NAVY, True, maxh=24)

def page(num, title):
    box(M, 29, 27, 3, TEAL, radius=0)
    text("BadFU Defend  /  연구 미팅", M+37, 23, 270, 8.5, 13, MUTED, maxh=16)
    text("2026.10.08", W-M-80, 23, 80, 8.5, 13, MUTED, maxh=16, align=2)
    line(M, H-40, W-M, H-40)
    text(title, M, H-31, CW-70, 8, 12, MUTED, maxh=13)
    # Footer uses direct text because its baseline is below the body safety guard.
    c.setFont("KoreanBold", 8.5)
    c.setFillColor(TEAL)
    c.drawRightString(W-M, 20, f"{num} / 3")

def footer_safe_page(num, title):
    # Footer text is drawn directly, leaving a strict 40pt body exclusion zone.
    box(M, 29, 27, 3, TEAL, radius=0)
    text("BadFU Defend  /  연구 미팅", M+37, 23, 270, 8.5, 13, MUTED, maxh=16)
    text("2026.10.08", W-M-80, 23, 80, 8.5, 13, MUTED, maxh=16, align=2)
    line(M, H-40, W-M, H-40)
    c.setFont("Korean", 8)
    c.setFillColor(MUTED)
    c.drawString(M, 20, title)
    c.setFont("KoreanBold", 8.5)
    c.setFillColor(TEAL)
    c.drawRightString(W-M, 20, f"{num} / 3")

def arrow(x, y):
    line(x-6, y, x+6, y, TEAL, 1.2)
    line(x+2, y-3, x+6, y, TEAL, 1.2)
    line(x+2, y+3, x+6, y, TEAL, 1.2)

# PAGE 1: purpose and evidence
footer_safe_page(1, "연구 목표와 현재 근거")
text("삭제를 이행하면서<br/>백도어를 정화하기", M, 64, CW, 27, 36, NAVY, True, maxh=76)
text("교수님 미팅 핵심 요약  |  가능성 관측 → 실제 FU 검증", M, 148, CW, 10, 16, MUTED, maxh=18)

box(M, 184, CW, 91, PALE)
text("탐지용 언러닝 0회.<br/>실제 FU 안에서 선택적 정화.", M+17, 197, CW-34,
     17.5, 25, TEAL, True, maxh=52)
text("연구 목표: 요청된 삭제를 수행하며 정상 성능과 공격 억제를 함께 평가", M+17, 254,
     CW-34, 8.8, 13, INK, maxh=15)

heading("문제는 삭제 이후 드러나는 백도어", 295, "01")
text("학습 중에는 백도어 기여와 위장 기여가 서로 상쇄될 수 있다.<br/>위장 쪽을 삭제하면 잠복한 공격이 활성화될 수 있다. [1, 2]",
     M, 326, CW, 10.3, 16, maxh=34)
gap, bw = 25, (CW-50)/3
for idx, (a,b) in enumerate([
    ("학습 중", "백도어 + 위장 → 잠복"),
    ("실제 삭제 요청", "위장 기여 제거"),
    ("삭제 후 위험", "잔존 백도어 활성화")]):
    x=M+idx*(bw+gap)
    box(x, 373, bw, 51, GRAY)
    text(a, x+10, 382, bw-20, 9.6, 14, NAVY, True, maxh=16, align=1)
    text(b, x+7, 401, bw-14, 8.3, 12, MUTED, maxh=14, align=1)
    if idx<2: arrow(x+bw+gap/2, 399)

heading("현재 결론은 가능성 확인 단계", 447, "02")
xcols=[M, M+96, M+298]
widths=[96,202,CW-298]
box(M, 477, CW, 26, NAVY, radius=4)
for x,w,s in zip(xcols,widths,["항목","확보한 근거","해석"]):
    text(s,x+10,483,w-20,8.8,13,WHITE,True,maxh=15)
rows=[
    ("근사 정화", "ACC 2.06~8.39%p 증가<br/>ASR 12.29~26.07%p 감소", "기여 차감 모델의<br/>예비 개선"),
    ("정상 대조", "정상 8명 중 5명 오탐", "요청 단위 FPR과<br/>구분해야 함"),
    ("과거 실제 FU", "ASR 63.70% → 60.20%", "분할·실행 대응<br/>재검증 필요"),
    ("새 GPU 코드", "15개 테스트·CPU 통합 통과", "GPU 실데이터<br/>성능 결과 대기"),
]
y=503
for idx,row in enumerate(rows):
    if idx%2==0: box(M,y,CW,43,GRAY,radius=0)
    for col,(s,x,w) in enumerate(zip(row,xcols,widths)):
        text(s,x+10,y+8,w-20,9.0,13.8,NAVY if col==0 else INK,
             col==0,maxh=29)
    line(M,y+43,W-M,y+43)
    y+=43
text("과거 수치는 FUBA 저장 로그의 관측값이다. 5개 실행을 독립 seed 5개로 보지 않는다.<br/>새 GPU 검증 결과와 구분하며, 낮은 ASR만으로 망각을 입증할 수 없다.",
     M, 686, CW, 8.8, 13.5, MUTED, maxh=29)
box(M,729,CW,51,PALE)
text("<b>첫 적용 범위</b>  개별 업데이트가 보이는 클라이언트 분리형 공격.<br/>BadFU 공식 ul 예제를 먼저 검증한다. 논문 본문의 단일 클라이언트 내부 위장 샘플 삭제는 별도 확장이다. [1]",
     M+13,740,CW-26,9.0,14,INK,maxh=30)
c.showPage()

# PAGE 2: algorithm
footer_safe_page(2, "알고리즘과 정화 원칙")
text("알고리즘과 개입 원칙", M, 64, CW, 25, 34, NAVY, True, maxh=38)
text("삭제 요청자 q는 이미 알고 있다.<br/>q와 잔존 참여자의 관계를 검사하고, 실제 FU의 보정 기여에 개입한다.",
     M, 111, CW, 10.5, 16.5, MUTED, maxh=35)
steps=[
    ("01","업데이트 기록 정렬",
     "동일 글로벌 모델에서 시작한 로컬 차이를 사용한다. 집계 가중치를 반영하며, 최근 16라운드와 최소 공동 관측 3회를 기본으로 둔다."),
    ("02","요청자 중심 상쇄 탐지",
     "반대 방향·상쇄량·상대 크기의 평균에 반복성을 반영한다. 정상 기준을 넘지 않으면 무경보로 두며, 요청자 삭제는 계속 수행한다."),
    ("03","의심 부분공간 B 추정",
     "선택된 클래스 헤드 기록에 비중심 SVD를 적용한다. 에너지 90%, 최대 랭크 8. 탐지 점수에서 얻은 가중치 γ는 악성 확률이 아니다."),
    ("04","실제 FU 안에서 선택 감쇠",
     "q를 제외한 잔존자의 보정 업데이트 중 B 성분만 줄인다. 실제 FU 계산을 재사용하며, 탐지를 위해 별도 FU를 실행하지 않는다. [7]"),
    ("05","정상 피해 제한",
     "제거 노름과 깨끗한 검증 손실을 검사한다. 허용 범위를 넘으면 해당 단계의 정화만 취소한다. 이때 보안 위험은 미해결로 기록한다."),
]
y=166
for idx,(num,title,body) in enumerate(steps):
    if idx<4: line(M+15,y+28,M+15,y+68,LINE,1.3)
    box(M,y+1,30,27,TEAL if idx==3 else PALE,radius=7)
    text(num,M+3,y+5,24,9.5,14,WHITE if idx==3 else TEAL,True,maxh=16,align=1)
    text(title,M+43,y,CW-43,12.1,18,NAVY,True,maxh=20)
    text(body,M+43,y+24,CW-43,9.6,14.8,INK,maxh=31)
    y+=68

box(M,521,CW,84,NAVY)
text("핵심 정화식",M+16,532,CW-32,9.2,14,colors.HexColor("#9ED6D0"),True,maxh=16)
text("정화 모델 = 실제 FU 모델 - 강도 × 의심 성분",M+16,555,CW-32,
     13,19,WHITE,True,maxh=22)
text("의심 성분: 잔존 보정 기여를 탐지 가중치로 합한 뒤 B에 투영한 값",M+16,583,CW-32,
     8.5,13,WHITE,maxh=15)

cards=[
    ("삭제 대상 고정","탐지 결과가 바뀌어도<br/>요청자 q는 제외"),
    ("강도 0 검증","정화 없는 FU와<br/>동일 모델 해시"),
    ("망각 별도 검증","FU 수행 자체가<br/>망각 보장은 아님"),
]
gap=12; bw=(CW-2*gap)/3
for idx,(a,b) in enumerate(cards):
    x=M+idx*(bw+gap)
    box(x,625,bw,66,GRAY)
    text(a,x+12,636,bw-24,10.3,15,NAVY,True,maxh=17)
    text(b,x+12,659,bw-24,8.8,13.3,MUTED,maxh=28)

text("<b>현재 기본값</b>  정화 강도 ≤ 0.5 · 제거량 ≤ FU 갱신 노름의 25% 및 절대 노름 1.0<br/>정상 검증 손실 증가가 0.01을 초과하면 해당 단계 정화를 취소한다.",
     M,711,CW,8.9,14.2,INK,maxh=30)
text("<b>비용 가설</b>  요청자 중심 점수 계산 O(WNCd), 전체 쌍 O(WN²Cd).<br/>기록·I/O·SVD·정상 검증 비용도 남는다. 전체 비용 우위는 실측으로 검증한다.",
     M,757,CW,8.7,13.2,MUTED,maxh=28)
c.showPage()

# PAGE 3: evaluation and literature
footer_safe_page(3, "검증 계획과 선행연구")
text("무엇을 검증하고 논의할까", M, 64, CW, 25, 34, NAVY, True, maxh=38)
text("우선 실제 FU에서 정화 효과를 확인한다. 이후 탐지 개선과 적용 범위를 넓힌다.",
     M,111,CW,10,16,MUTED,maxh=20)
heading("GPU 결과를 읽는 순서",146,"03")
checks=[
    ("1  조건 확인", "삭제 전후 공격 활성화와 none/zero 모델 해시 일치를 확인한다."),
    ("2  정화 효과", "none·oracle 후보·detected·조건을 맞춘 random을 비교한다.<br/>oracle도 완벽한 백도어 정답 공간은 아니다."),
    ("3  정상·삭제", "정상 요청의 오탐과 정확도 손실을 확인한다. 요청자 제외 retrain을<br/>기준으로 망각 진단을 보강한다. 삭제 데이터 손실만으로는 부족하다."),
    ("4  지속·비용", "정상 후속 FL·지속 공격·공격 재개를 나눠 최대 ASR과 궤적을 본다.<br/>탐지·기저·검증을 포함한 전체 추가 비용을 측정한다."),
]
y=177
for idx,(a,b) in enumerate(checks):
    if idx%2==0: box(M,y,CW,47,GRAY,radius=0)
    text(a,M+10,y+11,90,9.5,15,TEAL,True,maxh=18)
    text(b,M+108,y+8,CW-118,9.3,14.5,INK,maxh=31)
    line(M,y+47,W-M,y+47)
    y+=47
text("기본: CIFAR-10·ResNet18 / FL 40라운드 / 평가 seed 3개 / 후속 FL 10라운드.<br/>FU는 FedEraser 계열의 통제 구현이며 원본의 완전 재현은 아니다.<br/>ASR 10~15%는 후보 목표이며 안전 기준이 아니다. 장기 지속성은 추가 검증한다.",
     M,376,CW,8.8,13.7,MUTED,maxh=43)

heading("차별점은 삭제 이행과 정화의 결합",430,"04")
related=[
    ("BadFU / FUBA [1, 2]","위장 삭제로 활성화되는 공격. 분리형과 단일 클라이언트 설정을 구분."),
    ("MASA [3]","개별 언러닝으로 탐지. 제안은 탐지용 언러닝을 추가하지 않음."),
    ("탐지용 FU 없는 방어 [4, 8]","FoolsGold·FLDetector·FLAME도 존재. ‘시뮬레이션 없음’만으로 새롭지는 않음."),
    ("Malicious Forgetting [5]","FU 공격과 방향 부분공간 탐지의 근접 연구. 본문 세부 대조가 남아 있음."),
    ("Neurotoxin [6]","지속적인 백도어의 사례. 후속 FL이 자동으로 공격을 지운다고 가정하지 않음."),
]
y=461
for a,b in related:
    text(a,M,y,159,9.1,14,NAVY,True,maxh=29)
    text(b,M+170,y,CW-170,9.2,14,INK,maxh=29)
    line(M,y+33,W-M,y+33)
    y+=37

box(M,659,CW,65,PALE)
text("교수님께 받을 결정",M+13,670,CW-26,10.8,16,TEAL,True,maxh=18)
text("① 첫 논문을 분리형 공격으로 한정할지  ② 깨끗한 검증 데이터 접근을 허용할지<br/>③ 정상 성능·삭제 효과의 허용 손실과 성공 기준을 어떻게 정할지",
     M+13,694,CW-26,9.0,14,INK,maxh=29)

refs=[
    ("[1] BadFU · arXiv v1","https://arxiv.org/html/2508.15541v1"),
    ("[2] FUBA · IEEE TAI","https://ieeexplore.ieee.org/document/11231135/"),
    ("[3] MASA · WACV 2025","https://arxiv.org/html/2411.01040v1"),
    ("[4] FLAME · USENIX 2022","https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen"),
    ("[5] Malicious Forgetting · 저자 초록","https://wenwei-zhao.github.io/publication/conference-paper/3/"),
    ("[6] Neurotoxin · ICML 2022","https://proceedings.mlr.press/v162/zhang22w.html"),
    ("[7] FedEraser · IWQOS 2021","https://www.cs.sfu.ca/~jcliu/Papers/FedEraser21.pdf"),
    ("[8] 상세 근거와 논문 13편","https://github.com/sijoon-sung/BadFU_defend/blob/codex/request-aware-purification/docs/research/LITERATURE_REVIEW_2026-10-08.md"),
]
text("원문 링크  ·  항목을 클릭하면 자료가 열립니다",M,739,CW,8,12,MUTED,maxh=14)
for idx,(label,url) in enumerate(refs):
    row,col=divmod(idx,2)
    x=M+col*(CW/2+4); yy=758+row*10.7
    text(f'<link href="{escape(url)}" color="#047F7C">{escape(label)}</link>',
         x,yy,CW/2-9,7.6,10.2,TEAL,maxh=10.5)
c.showPage()
c.save()
print(OUT)
