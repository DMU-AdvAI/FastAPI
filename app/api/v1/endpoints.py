from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test():
    """
    테스트 API 엔드포인트
    - return: 테스트 성공 메시지
    - parameter : null
    """
    return {"message" : "테스트 성공"}