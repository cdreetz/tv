"use client"

import { SidebarLeft } from "@/components/sidebar-left"
import { SidebarRight } from "@/components/sidebar-right"
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/ui/sidebar"
import PhaseOne from "@/components/phase-one"
import PhaseTwo from "@/components/phase-two"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { BarChart3, Building2 } from "lucide-react"

export default function Page() {
  const [currentMode, setCurrentMode] = useState<'comparison' | 'similar'>('comparison')

  return (
    <SidebarProvider>
      <SidebarLeft />
        <SidebarInset>
          {/* Mode Toggle Header */}
          <div className="flex items-center justify-center gap-2 p-4 border-b bg-background">
            <Button
              variant={currentMode === 'comparison' ? 'default' : 'outline'}
              onClick={() => setCurrentMode('comparison')}
              className="flex items-center gap-2"
            >
              <BarChart3 className="h-4 w-4" />
              Comparison Reports
            </Button>
            <Button
              variant={currentMode === 'similar' ? 'default' : 'outline'}
              onClick={() => setCurrentMode('similar')}
              className="flex items-center gap-2"
            >
              <Building2 className="h-4 w-4" />
              Similar Companies
            </Button>
          </div>
          
          {/* Render the appropriate component based on mode */}
          {currentMode === 'comparison' ? <PhaseOne /> : <PhaseTwo />}
        </SidebarInset>
      <SidebarRight />
    </SidebarProvider>
  )
}
