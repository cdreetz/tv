#!/usr/bin/env python3
"""
Market Report Pipeline API

A FastAPI backend that provides streaming capabilities for market report generation.
Supports Server-Sent Events and REST endpoints.
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import streaming classes
from .report_gen_stream import MarketReportGenerator
from .common import PipelineStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global generator instance
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the report generator"""
    global generator
    try:
        logger.info("Initializing MarketReportGenerator...")
        generator = MarketReportGenerator()
        logger.info("MarketReportGenerator initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize MarketReportGenerator: {e}")
        raise
    finally:
        logger.info("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Market Report Pipeline API",
    description="Real-time market report generation with streaming progress updates",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CompanyRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company to analyze")

class ComparisonRequest(BaseModel):
    companies: List[str] = Field(..., description="List of company names to compare", min_items=2)

class ReportResponse(BaseModel):
    success: bool
    report: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class CompaniesResponse(BaseModel):
    companies: List[str]
    total_count: int

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Market Report Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "streaming": {
                "/sse/report/{company_name}": "Stream single company report",
                "/sse/comparison": "Stream comparison report"
            },
            "rest": {
                "/api/report": "Generate single company report",
                "/api/comparison": "Generate comparison report",
                "/api/companies": "Get available companies"
            }
        }
    }

# Server-Sent Events endpoints
@app.get("/sse/report/{company_name}")
async def sse_single_report(company_name: str):
    """Stream single company report progress"""
    
    async def event_stream():
        try:
            async for status in generator.generate_single_company_report_stream(company_name):
                yield f"data: {json.dumps(status.to_dict())}\n\n"
                if status.stage in ["completion", "error"]:
                    break
        except Exception as e:
            error_status = PipelineStatus(
                stage="error",
                message=f"Internal server error: {str(e)}",
                progress=0.0,
                error=str(e)
            )
            yield f"data: {json.dumps(error_status.to_dict())}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/sse/comparison")
async def sse_comparison_report(companies: str):
    """Stream comparison report progress"""
    
    company_list = [c.strip() for c in companies.split(",") if c.strip()]
    
    if len(company_list) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least 2 companies separated by commas"
        )
    
    async def event_stream():
        try:
            async for status in generator.generate_comparison_report_stream(company_list):
                yield f"data: {json.dumps(status.to_dict())}\n\n"
                if status.stage in ["completion", "error"]:
                    break
        except Exception as e:
            error_status = PipelineStatus(
                stage="error",
                message=f"Internal server error: {str(e)}",
                progress=0.0,
                error=str(e)
            )
            yield f"data: {json.dumps(error_status.to_dict())}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

# REST endpoints
@app.post("/api/report", response_model=ReportResponse)
async def api_single_report(request: CompanyRequest):
    """Generate single company report"""
    try:
        report = await generator.generate_single_company_report(request.company_name)
        return ReportResponse(success=True, report=report)
    except Exception as e:
        return ReportResponse(success=False, error=str(e))

@app.post("/api/comparison", response_model=ReportResponse)
async def api_comparison_report(request: ComparisonRequest):
    """Generate comparison report"""
    try:
        report = await generator.generate_comparison_report(request.companies)
        return ReportResponse(success=True, report=report)
    except Exception as e:
        return ReportResponse(success=False, error=str(e))

@app.get("/api/companies", response_model=CompaniesResponse)
async def get_available_companies():
    """Get list of available companies"""
    try:
        companies = generator.company_names
        return CompaniesResponse(
            companies=sorted(companies),
            total_count=len(companies)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "generator_initialized": generator is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Market Report Pipeline API...")
    print("📡 SSE endpoints: /sse/report/{company} and /sse/comparison")
    print("🔄 REST endpoints: /api/report and /api/comparison")
    print("📚 API docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "src.stream.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
