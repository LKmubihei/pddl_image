(define (problem picture_53)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_pump blue_pump green_pump - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_pump)
        (clear blue_pump)
        (clear green_pump)
        (part_at red_pump table)
        (part_at blue_pump table)
        (part_at green_pump table)
    )
    
    (:goal
        (and
            (part_at red_pump pump_placement)
        )
    )
)
