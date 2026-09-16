import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Vehicle Damage Detection & Assessment Application")
    print("Web Dashboard: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
