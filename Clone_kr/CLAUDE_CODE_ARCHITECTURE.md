# LLMunix와 Claude Code 아키텍처 통합

## 개요

이 문서는 LLMunix가 Claude Code의 네이티브 아키텍처, 특히 서브 에이전트 시스템과 어떻게 통합되는지에 대한 상세한 설명을 제공합니다. LLMunix는 "순수 마크다운 운영체제"로서, 순수 마크다운 철학을 유지하면서 창발적 지능 시스템을 만들기 위해 Claude Code의 기본 기능을 영리하게 활용합니다.

## 아키텍처 통합

### 순수 마크다운 추상화

LLMunix는 "모든 것이 마크다운 문서로 정의된 에이전트 또는 도구"인 운영체제로 표현됩니다. 이것은 다음을 통해 달성됩니다:

1. **마크다운 정의 레이어**: 모든 구성요소(에이전트 및 도구)가 YAML 프론트매터가 있는 마크다운 문서로 정의됨
2. **Claude Code 런타임 레이어**: 이러한 마크다운 정의가 Claude Code의 네이티브 도구 및 서브 에이전트에 매핑됨
3. **초기화 프로세스**: 설정 스크립트(`setup_agents.ps1`/`setup_agents.sh`)가 에이전트 마크다운 파일을 Claude Code가 발견하는 `.claude/agents/` 디렉토리에 복사

### 서브 에이전트 아키텍처

#### LLMunix 에이전트가 Claude Code 서브 에이전트에 매핑되는 방법

LLMunix가 "부팅"될 때, 마크다운으로 정의된 에이전트와 Claude Code의 서브 에이전트 시스템 간의 매핑을 설정합니다:

1. **에이전트 등록**: LLMunix 에이전트 마크다운 파일이 Claude Code가 서브 에이전트로 발견할 수 있는 `.claude/agents/` 디렉토리에 복사됨
2. **메타데이터 매핑**: 각 마크다운 파일의 YAML 프론트매터가 Claude Code의 서브 에이전트 시스템에 필요한 구성을 제공:
   - `name`: 서브 에이전트 식별자 정의
   - `description`: Claude Code가 이 에이전트를 언제 사용할지 이해하는 데 도움
   - `tools`: 서브 에이전트가 접근할 수 있는 도구 지정

3. **실행 흐름**: system-agent가 다른 에이전트에게 위임할 때, Claude Code의 Task 도구를 사용하여 서브 에이전트로 호출

#### Claude Code의 서브 에이전트 시스템

Claude Code의 서브 에이전트 시스템은 LLMunix가 활용하는 여러 핵심 기능을 제공합니다:

1. **컨텍스트 격리**: 각 서브 에이전트가 자체 컨텍스트 창에서 작동하여 메인 대화의 오염 방지
2. **특화된 전문성**: 서브 에이전트가 특정 도메인에 대한 상세한 지시로 미세 조정 가능
3. **도구 접근 제어**: 각 서브 에이전트에게 특정 도구에 대한 접근만 부여 가능
4. **독립적 작업 실행**: 서브 에이전트가 독립적으로 작업하고 호출자에게 결과 반환

### 런타임 실행 분석

`llmunix execute: "https://huggingface.co/blog에서 실시간 콘텐츠를 가져와 연구 요약 작성"`과 같은 명령을 실행할 때 다음이 발생합니다:

1. **System Agent 호출**: Claude Code의 내장 system-agent 서브 에이전트가 Task 도구를 통해 호출됨
2. **상태 초기화**: system-agent가 작업 공간 상태 디렉토리 구조를 생성하고 초기화
3. **서브 에이전트 위임**: system-agent가 Task 도구를 사용하여 특화된 작업 위임:
   - `WebFetch`: Hugging Face 블로그 콘텐츠 검색
   - `real-summarization-agent`: 콘텐츠 분석 및 요약
   - `Write`: 최종 출력 저장

4. **상태 관리**: 실행 전체에서 모듈형 상태 파일이 원자적으로 업데이트됨
5. **메모리 기록**: 경험이 구조화된 메타데이터와 함께 메모리 로그에 기록됨

## LLMunix의 핵심 서브 에이전트

LLMunix는 Claude Code의 서브 에이전트 시스템에 매핑되는 여러 특화된 서브 에이전트를 정의합니다:

1. **system-agent**: 작업을 위임하고 시스템 상태를 관리하는 핵심 오케스트레이터
   - **도구**: Read, Write, Glob, Grep, Bash, WebFetch, Task
   - **역할**: 고수준 계획, 상태 관리, 서브 에이전트 조율

2. **real-summarization-agent**: 콘텐츠 분석 및 요약 전문
   - **도구**: Read, Write, WebFetch
   - **역할**: 콘텐츠 소스 처리, 핵심 정보 추출, 구조화된 요약 생성

3. **memory-analysis-agent**: 과거 실행 패턴 분석
   - **도구**: Read, Grep, Bash
   - **역할**: 메모리 로그 쿼리, 패턴 식별, 향후 작업을 위한 인사이트 제공

4. **market-analyst-agent**, **intelligence-briefing-agent**, **content-writer-agent** 등
   - **역할**: 특화된 지식과 기능을 가진 도메인 특화 작업

## 시스템 vs. LLMunix 에이전트

LLMunix가 작업을 실행할 때:

1. **Claude Code의 네이티브 system-agent 사용**: 오케스트레이션이 LLMunix에서 정의한 커스텀 에이전트가 아닌 Claude Code의 내장 system-agent 서브 에이전트를 통해 발생

2. **그러나 LLMunix의 마크다운 정의와 함께**: 에이전트의 동작이 `.claude/agents/SystemAgent.md`에 복사된 `SystemAgent.md`의 마크다운 정의에 의해 안내됨

3. **하이브리드 실행 모델**: 실행이 다음을 결합:
   - 격리 및 도구 접근을 위한 Claude Code의 네이티브 서브 에이전트 아키텍처
   - LLMunix의 마크다운으로 정의된 동작 명세
   - 작업 공간 디렉토리의 파일 시스템 작업을 통한 상태 관리

## 기술적 구현 세부사항

### 에이전트 등록 프로세스

초기화 중:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\setup_agents.ps1

# Unix/Linux/Mac
./setup_agents.sh
```

이 스크립트들이 수행하는 작업:
1. `.claude/agents/` 디렉토리가 없으면 생성
2. `system/agents/` 및 `components/agents/`의 모든 에이전트 마크다운 파일을 `.claude/agents/`에 복사
3. 이것이 Claude Code의 서브 에이전트 시스템에서 발견 가능하게 만듦

### 에이전트 정의 형식

각 에이전트는 다음 구조의 마크다운 파일로 정의됩니다:

```markdown
---
name: agent-name
description: 이 에이전트를 언제 사용해야 하는지
tools: tool1, tool2, tool3
---

# Agent Name: 목적

에이전트에 대한 상세한 지시...
```

이 형식은 두 가지 목적을 제공합니다:
- 사람이 읽을 수 있는 문서
- Claude Code의 서브 에이전트 시스템을 위한 머신 실행 가능 명세

### 작업 위임

system-agent는 Task 도구를 사용하여 다른 에이전트에게 작업을 위임합니다:

```
Action: Task
Parameters:
  description: "작업 설명"
  prompt: "상세한 작업 지시"
  subagent_type: "specialized-agent-name"
```

### 상태 관리

순수 마크다운 특성이 파일 시스템 작업을 통해 유지됩니다:

1. **상태 디렉토리**: `workspace/state/`에 모듈형 파일 포함 (plan.md, context.md, variables.json 등)
2. **원자적 업데이트**: 각 파일이 깔끔한 상태 전환을 유지하기 위해 독립적으로 업데이트됨
3. **메모리 지속성**: `system/memory_log.md`에 향후 참조를 위한 구조화된 경험 저장

## 이 아키텍처의 이점

1. **Claude Code의 강력함 활용**: 강력한 내장 도구 및 서브 에이전트 격리에 대한 접근
2. **순수 마크다운 철학**: 사람이 읽을 수 있고 버전 제어 가능한 시스템 정의 유지
3. **동적 진화**: 런타임에 새 에이전트 마크다운 파일을 생성하는 능력
4. **지각적 상태 관리**: 실행 이벤트에 따라 진화하는 동작 제약
5. **메모리 기반 학습**: 현재 작업 실행에 영향을 미치는 과거 경험

## 실용적인 예시: Hugging Face 블로그 연구

`llmunix execute: "https://huggingface.co/blog에서 실시간 콘텐츠를 가져와 연구 요약 작성"`을 실행했을 때:

1. **Claude Code system-agent 서브 에이전트**가 SystemAgent.md의 지시와 함께 호출됨
2. **워크플로우 오케스트레이션** 수행:
   - 작업 공간 상태 디렉토리 생성
   - WebFetch를 사용하여 블로그 콘텐츠 검색
   - real-summarization-agent에게 콘텐츠 분석 위임
   - 작업 공간 디렉토리에 최종 연구 요약 작성

3. **실행 전체에서**:
   - 상태 파일이 원자적으로 업데이트됨
   - 메모리 로그가 기록됨
   - 실행 이벤트에 따라 제약이 진화함

## 결론

LLMunix는 AI 시스템 설계에 대한 혁신적인 접근 방식을 보여줍니다:

1. **추상화 레이어 생성**: 마크다운을 정의 언어로 사용
2. **기존 인프라 활용**: Claude Code의 네이티브 기능에 매핑
3. **창발적 지능 활성화**: 특화된 에이전트, 메모리 시스템, 적응적 제약의 조합을 통해

이 아키텍처는 LLMunix가 Claude Code의 강력한 도구 및 서브 에이전트 시스템의 이점을 누리면서 "순수 마크다운 운영체제" 철학을 유지할 수 있게 합니다.
